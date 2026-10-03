#!/usr/bin/env python3
"""
podcast_pipeline.py

For each episode in the RSS feed without a complete set of transcripts:
  1. Download the audio to a working directory (resumable)
  2. Transcribe it with faster-whisper
  3. Write srt/vtt/txt atomically (txt last, since it marks the episode done)
  4. Delete the audio
Then commit everything new or changed under txt/, srt/, vtt/ and push.

Resilience notes:
  - Commit and push run even if the loop crashes or is interrupted (Ctrl-C).
  - The commit sweeps up anything a previous failed run left uncommitted, so
    rerunning with nothing to transcribe doubles as a "sync to GitHub" step.
  - An episode only counts as done when all three files exist; half-finished
    episodes get redone on the next run.
  - Outputs are written to a temp file and renamed, so a crash never leaves a
    truncated transcript behind.
  - Downloads resume correctly after mid-stream failures, and fall back to a
    full download if the server ignores Range requests.

Layout (relative to this script):
  ../txt/ ../srt/ ../vtt/   committed
  ../working/               temp audio + partial writes (.gitignored)

Flags:
  --no-commit   transcribe only, leave git alone
  --no-push     commit but don't push
"""

import argparse
import email.utils
import glob
import hashlib
import os
import random
import re
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse

import feedparser
import requests
from dateutil import parser as dtparser
from tqdm import tqdm
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RSS_URL = "https://anchor.fm/s/45580f58/podcast/rss"

# Whisper
WHISPER_MODEL = "medium"
BEAM_SIZE = 5
MIN_SILENCE_MS = 300
LANGUAGE = "en"

# Network
MAX_RETRIES = 8
BASE_SLEEP = 1.0
MAX_SLEEP = 120.0
JITTER = (0.75, 1.25)
BETWEEN_DOWNLOAD_SLEEP = (0.8, 2.2)
TIMEOUT = (10, 60)
USER_AGENT = "PodcastBulkDL/1.2 (+https://example.com)"
RETRY_STATUSES = {429, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Paths (all absolute, derived from this file's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
WORKING_DIR = os.path.join(TRANSCRIPTS_DIR, "working")
TXT_DIR = os.path.join(TRANSCRIPTS_DIR, "txt")
SRT_DIR = os.path.join(TRANSCRIPTS_DIR, "srt")
VTT_DIR = os.path.join(TRANSCRIPTS_DIR, "vtt")

# Order matters: txt is written last because its presence used to be the
# "done" marker, and keeping it last means older checks stay safe too.
OUTPUTS = ((SRT_DIR, ".srt"), (VTT_DIR, ".vtt"), (TXT_DIR, ".txt"))
OUTPUT_DIRS = (TXT_DIR, SRT_DIR, VTT_DIR)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def slugify(text: str, max_len: int = 120) -> str:
    text = unidecode(text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:max_len] or "episode"


def transcript_basename(title: str, published: str, audio_url: str) -> str:
    """Build the shared base name: YYYY-MM-DD-slug-hash"""
    date_prefix = f"{published}-" if published else ""
    return f"{date_prefix}{slugify(title)}-{short_hash(audio_url)}"


def existing_base(audio_url: str) -> str | None:
    """Base name of any output already on disk for this URL.

    Reusing it means a partially finished episode gets completed under its
    original name, even if the episode title changed in the feed since.
    """
    h = short_hash(audio_url)
    for d, ext in OUTPUTS:
        hits = glob.glob(os.path.join(d, f"*{h}{ext}"))
        if hits:
            return os.path.splitext(os.path.basename(hits[0]))[0]
    return None


def is_complete(audio_url: str) -> bool:
    """True only if txt, srt, and vtt all exist for this URL."""
    h = short_hash(audio_url)
    return all(glob.glob(os.path.join(d, f"*{h}{ext}")) for d, ext in OUTPUTS)


def remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def backoff(attempt: int) -> float:
    return min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt)) * random.uniform(*JITTER)


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header, which can be seconds or an HTTP date."""
    if not value:
        return None
    try:
        return min(MAX_SLEEP, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return min(MAX_SLEEP, max(0.0, dt.timestamp() - time.time()))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def fetch_bytes(session: requests.Session, url: str) -> bytes:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code in RETRY_STATUSES:
                wait = retry_after_seconds(r.headers.get("Retry-After")) or backoff(attempt)
                print(f"  Got {r.status_code} fetching feed, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.content
        except requests.HTTPError:
            raise  # 4xx and friends won't get better by retrying
        except requests.RequestException as ex:
            if attempt == MAX_RETRIES:
                raise
            wait = backoff(attempt)
            print(f"  Network error fetching feed: {ex}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Gave up fetching {url}")


def parse_rss(session: requests.Session, url: str) -> list[dict]:
    feed = feedparser.parse(fetch_bytes(session, url))
    # feedparser sets `bozo` for harmless things like encoding mismatches, so
    # only treat it as fatal if nothing usable came back.
    if not feed.entries:
        raise RuntimeError(f"RSS feed has no entries ({feed.get('bozo_exception')})")

    items = []
    for entry in feed.entries:
        title = getattr(entry, "title", "Untitled")
        pub = getattr(entry, "published", "") or getattr(entry, "pubDate", "")
        try:
            pub_iso = dtparser.parse(pub).date().isoformat() if pub else ""
        except Exception:
            pub_iso = ""

        audio = ""
        for enc in getattr(entry, "enclosures", []):
            if enc.get("href"):
                audio = enc["href"]
                break
        if not audio:
            for link in getattr(entry, "links", []):
                href = link.get("href", "")
                if href.lower().endswith((".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac")):
                    audio = href
                    break
        if audio:
            items.append({"title": title, "published": pub_iso, "audio_url": audio})
    return items


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def content_length(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.ok:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except requests.RequestException:
        pass
    return None


def download_episode(session: requests.Session, url: str, dest: str) -> None:
    """Download to dest via dest.part, resuming where a previous attempt stopped."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return  # left over from a run that died before transcribing

    tmp = dest + ".part"
    total = content_length(session, url)
    attempt = 0

    while True:
        # Recompute the offset every attempt, so a resume after a mid-stream
        # failure asks for the right byte range.
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}

        try:
            with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                if r.status_code in RETRY_STATUSES:
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise RuntimeError(f"Max retries reached ({r.status_code})")
                    wait = retry_after_seconds(r.headers.get("Retry-After")) or backoff(attempt)
                    print(f"  Got {r.status_code}, retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                if r.status_code == 416:
                    if total is None or have >= total:
                        break  # server says we already have everything
                    # The partial doesn't line up with the server; start over.
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise RuntimeError("Server keeps rejecting the resume range")
                    remove_quietly(tmp)
                    continue

                r.raise_for_status()

                mode = "ab"
                if have and r.status_code != 206:
                    # Server ignored Range and sent the whole file. Appending
                    # would corrupt it, so start the file over.
                    mode, have = "wb", 0

                with open(tmp, mode) as fh, tqdm(
                    total=(total - have) if total else None,
                    unit="B",
                    unit_scale=True,
                    desc=f"  ↓ {os.path.basename(dest)[:40]}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            fh.write(chunk)
                            pbar.update(len(chunk))

            if total is None or os.path.getsize(tmp) >= total:
                break
            # Stream ended early without an exception. Loop around and resume.
            attempt += 1
            if attempt > MAX_RETRIES:
                raise RuntimeError("Download keeps ending early")
            print("  Download ended early, resuming...")

        except requests.HTTPError:
            raise
        except requests.RequestException as ex:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = backoff(attempt)
            print(f"  Network error: {ex}. Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)

    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError("Downloaded file is empty")
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Transcript rendering and writing
# ---------------------------------------------------------------------------

def ts(seconds, srt: bool = False) -> str:
    ms = int(round((seconds or 0.0) * 1000))
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, mmm = divmod(rem, 1_000)
    sep = "," if srt else "."
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{mmm:03d}"


def nonempty(segments):
    for s in segments:
        text = (s.text or "").strip()
        if text:
            yield s, text


def render_txt(segments) -> str:
    return " ".join(text for _s, text in nonempty(segments)) + "\n"


def render_srt(segments) -> str:
    # Number only the cues actually written, so there are no gaps.
    blocks = [
        f"{i}\n{ts(s.start, srt=True)} --> {ts(s.end, srt=True)}\n{text}\n"
        for i, (s, text) in enumerate(nonempty(segments), 1)
    ]
    return "\n".join(blocks)


def render_vtt(segments) -> str:
    blocks = [f"{ts(s.start)} --> {ts(s.end)}\n{text}\n" for s, text in nonempty(segments)]
    return "WEBVTT\n\n" + "\n".join(blocks)


RENDERERS = {".txt": render_txt, ".srt": render_srt, ".vtt": render_vtt}


def write_atomic(path: str, content: str) -> None:
    """Write to a temp file in working/ (gitignored), then rename into place."""
    tmp = os.path.join(WORKING_DIR, os.path.basename(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Per-episode work
# ---------------------------------------------------------------------------

def process_episode(model, session: requests.Session, item: dict) -> str:
    url = item["audio_url"]
    base = existing_base(url) or transcript_basename(item["title"], item["published"], url)
    ext = os.path.splitext(urlparse(url).path)[1].lower() or ".mp3"
    audio_path = os.path.join(WORKING_DIR, base + ext)

    print("  Downloading...")
    download_episode(session, url, audio_path)

    print("  Transcribing...")
    try:
        segments, _info = model.transcribe(
            audio_path,
            beam_size=BEAM_SIZE,
            language=LANGUAGE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": MIN_SILENCE_MS},
            word_timestamps=False,
        )
        segs = list(segments)  # transcription actually happens here
    except Exception:
        # Most likely a bad download; remove it so the next run fetches fresh.
        remove_quietly(audio_path)
        raise

    if not any((s.text or "").strip() for s in segs):
        remove_quietly(audio_path)
        raise RuntimeError("Transcription produced no text")

    for d, ext_out in OUTPUTS:
        write_atomic(os.path.join(d, base + ext_out), RENDERERS[ext_out](segs))

    remove_quietly(audio_path)
    return base


def transcribe_pending() -> list[str]:
    """Process everything that isn't complete. Returns titles that failed."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print("Fetching RSS feed...")
    items = parse_rss(session, RSS_URL)
    pending = [it for it in items if not is_complete(it["audio_url"])]

    if not pending:
        print("All episodes already transcribed.")
        return []

    print(f"{len(pending)} episode(s) to process.")
    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    from faster_whisper import WhisperModel  # only pay the import cost when needed
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

    failures = []
    for i, item in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] {item['title']}")
        if i > 1:
            time.sleep(random.uniform(*BETWEEN_DOWNLOAD_SLEEP))
        try:
            base = process_episode(model, session, item)
            print(f"  ✓ {base}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures.append(item["title"])
    return failures


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def find_repo_root(start: str) -> str:
    path = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError("Could not find a git repository root")
        path = parent


def run_git(args: list[str], repo_root: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True, text=True, capture_output=True
    )


def commit_outputs(repo_root: str) -> bool:
    """Stage and commit everything new or changed in the output dirs.

    Adding the directories (not a list of files from this run) is what sweeps
    up anything an earlier crashed run left behind.
    """
    rel_dirs = [os.path.relpath(d, repo_root) for d in OUTPUT_DIRS]
    run_git(["add", "--", *rel_dirs], repo_root)
    changed = run_git(
        ["diff", "--cached", "--name-only", "--", *rel_dirs], repo_root
    ).stdout.splitlines()
    if not changed:
        return False

    bases = sorted({os.path.splitext(os.path.basename(p))[0] for p in changed})
    subject = f"Add transcripts for {len(bases)} episode(s)"
    body = "Transcribed episodes:\n" + "\n".join(f"  - {b}" for b in bases)
    # The pathspec limits the commit to transcript dirs, even if you happen to
    # have other changes staged elsewhere in the repo.
    run_git(["commit", "-m", f"{subject}\n\n{body}", "--", *rel_dirs], repo_root)
    print(f"Committed {len(changed)} file(s) for {len(bases)} episode(s).")
    return True


def sync_git(repo_root: str, push: bool) -> int:
    print("\nSyncing with git...")
    try:
        if not commit_outputs(repo_root):
            print("No transcript changes to commit.")
    except subprocess.CalledProcessError as exc:
        print(f"git {exc.cmd[1]} failed:\n{(exc.stderr or exc.stdout or '').strip()}")
        return 1

    if not push:
        return 0

    # Always push, even with no new commit, so earlier unpushed commits go up.
    r = subprocess.run(["git", "push"], cwd=repo_root, text=True, capture_output=True)
    if r.returncode != 0:
        print(f"git push failed:\n{r.stderr.strip()}")
        print("Your commits are safe locally. Fix the issue and rerun, or push by hand.")
        return 1
    print((r.stderr or r.stdout).strip() or "Pushed.")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download, transcribe, commit, and push podcast transcripts.")
    p.add_argument("--no-commit", action="store_true", help="transcribe only, leave git alone")
    p.add_argument("--no-push", action="store_true", help="commit but don't push")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for d in (WORKING_DIR, *OUTPUT_DIRS):
        os.makedirs(d, exist_ok=True)

    # Find the repo up front so a bad setup fails now, not after hours of work.
    repo_root = None if args.no_commit else find_repo_root(TRANSCRIPTS_DIR)

    exit_code = 0
    try:
        failures = transcribe_pending()
        if failures:
            exit_code = 1
            print(f"\n{len(failures)} episode(s) failed and will be retried next run:")
            for title in failures:
                print(f"  - {title}")
    except KeyboardInterrupt:
        print("\nInterrupted. Committing whatever finished.")
        exit_code = 130
    except Exception:
        traceback.print_exc()
        print("\nPipeline crashed. Committing whatever finished.")
        exit_code = 1
    finally:
        if repo_root:
            exit_code = sync_git(repo_root, push=not args.no_push) or exit_code

    try:
        os.rmdir(WORKING_DIR)  # only succeeds if empty
    except OSError:
        pass

    return exit_code


if __name__ == "__main__":
    code = main()
    # Skip normal interpreter teardown. CTranslate2 (under faster-whisper) can
    # crash while unloading the model, which makes a successful run look like
    # it failed. Everything important is on disk and in git by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
