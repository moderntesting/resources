#!/usr/bin/env python3
"""
podcast_pipeline.py

For each episode in the RSS feed that hasn't been transcribed yet:
  1. Download the mp3 to a temporary working directory
  2. Transcribe it with faster-whisper
  3. Write txt/srt/vtt to the appropriate sibling directories
  4. Delete the mp3
After all episodes are processed, commit the new transcript files to git.

Expected directory layout (relative to this script):
  ../txt/      ← plain-text transcripts  (committed)
  ../srt/      ← SubRip subtitles        (committed)
  ../vtt/      ← WebVTT subtitles        (committed)
  ../working/  ← temp mp3s + partials    (.gitignored)
"""

import glob
import hashlib
import os
import random
import re
import subprocess
import time
from urllib.parse import urlparse

import requests
import feedparser
from dateutil import parser as dtparser
from tqdm import tqdm
from unidecode import unidecode
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RSS_URL = "https://anchor.fm/s/45580f58/podcast/rss"

# Whisper
WHISPER_MODEL = "medium"
BEAM_SIZE = 5
MIN_SILENCE_MS = 300
LANGUAGE = "en"

# Download
MAX_RETRIES = 8
BASE_SLEEP = 1.0
MAX_SLEEP = 120.0
JITTER = (0.25, 1.25)
BETWEEN_DOWNLOAD_SLEEP = (0.8, 2.2)
TIMEOUT = (10, 60)
USER_AGENT = "PodcastBulkDL/1.1 (+https://example.com)"

# ---------------------------------------------------------------------------
# Paths  (all absolute, derived from this file's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
WORKING_DIR = os.path.join(TRANSCRIPTS_DIR, "working")
TXT_DIR = os.path.join(TRANSCRIPTS_DIR, "txt")
SRT_DIR = os.path.join(TRANSCRIPTS_DIR, "srt")
VTT_DIR = os.path.join(TRANSCRIPTS_DIR, "vtt")


# ---------------------------------------------------------------------------
# Naming helpers
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


def already_transcribed(audio_url: str) -> bool:
    """True if a .txt file with the URL's hash already exists."""
    h = short_hash(audio_url)
    return bool(glob.glob(os.path.join(TXT_DIR, f"*{h}.txt")))


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def parse_rss(url: str) -> list[dict]:
    feed = feedparser.parse(url)
    if feed.bozo:
        raise RuntimeError(f"RSS parse error: {feed.bozo_exception}")
    items = []
    for entry in feed.entries:
        title = getattr(entry, "title", "Untitled")
        pub = getattr(entry, "published", "") or getattr(entry, "pubDate", "")
        try:
            pub_iso = dtparser.parse(pub).date().isoformat() if pub else ""
        except Exception:
            pub_iso = pub

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

def _content_length(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.ok:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except requests.RequestException:
        pass
    return None


def download_episode(session: requests.Session, url: str, dest: str) -> None:
    tmp = dest + ".part"
    bytes_already = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    total_size = _content_length(session, url)

    headers = {"User-Agent": USER_AGENT}
    if bytes_already and total_size and bytes_already < total_size:
        headers["Range"] = f"bytes={bytes_already}-"

    attempt = 0
    with open(tmp, "ab") as fh:
        while True:
            try:
                with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                    if r.status_code in (429, 503, 502, 500):
                        ra = r.headers.get("Retry-After")
                        sleep_for = min(MAX_SLEEP, float(ra) if ra else BASE_SLEEP * (2 ** attempt))
                        attempt += 1
                        if attempt > MAX_RETRIES:
                            raise RuntimeError(f"Max retries reached for {url}")
                        print(f"  Rate limited ({r.status_code}), retrying in {sleep_for:.1f}s...")
                        time.sleep(sleep_for)
                        continue
                    if r.status_code == 416:
                        break  # range not satisfiable → already complete
                    r.raise_for_status()
                    display_total = (total_size - bytes_already) if (total_size and bytes_already) else total_size
                    with tqdm(total=display_total, unit="B", unit_scale=True,
                              desc=f"  ↓ {os.path.basename(dest)}") as pbar:
                        for chunk in r.iter_content(chunk_size=256 * 1024):
                            if chunk:
                                fh.write(chunk)
                                pbar.update(len(chunk))
                    break
            except requests.RequestException as ex:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                sleep_for = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
                print(f"  Network error: {ex}. Retry {attempt}/{MAX_RETRIES} in {sleep_for:.1f}s")
                time.sleep(sleep_for)

    if total_size and os.path.getsize(tmp) < total_size:
        raise RuntimeError(f"Incomplete download: {dest}")
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Transcript writers
# ---------------------------------------------------------------------------

def _ts(seconds, srt: bool = False) -> str:
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, mmm = divmod(rem, 1_000)
    sep = "," if srt else "."
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{mmm:03d}"


def write_txt(path: str, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            text = (s.text or "").strip()
            if text:
                f.write(text + " ")
        f.write("\n")


def write_srt(path: str, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            text = (s.text or "").strip()
            if not text:
                continue
            f.write(f"{i}\n{_ts(s.start, srt=True)} --> {_ts(s.end, srt=True)}\n{text}\n\n")


def write_vtt(path: str, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            f.write(f"{_ts(s.start)} --> {_ts(s.end)}\n{text}\n\n")


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


def git_commit(absolute_paths: list[str], repo_root: str) -> None:
    rel_paths = [os.path.relpath(p, repo_root) for p in absolute_paths]
    subprocess.run(["git", "add"] + rel_paths, cwd=repo_root, check=True)

    bases = sorted({os.path.splitext(os.path.basename(p))[0] for p in absolute_paths})
    subject = f"Add transcripts for {len(bases)} episode(s)"
    body = "Transcribed episodes:\n" + "\n".join(f"  - {b}" for b in bases)
    subprocess.run(["git", "commit", "-m", f"{subject}\n\n{body}"], cwd=repo_root, check=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    for d in (WORKING_DIR, TXT_DIR, SRT_DIR, VTT_DIR):
        os.makedirs(d, exist_ok=True)

    print("Fetching RSS feed...")
    all_items = parse_rss(RSS_URL)
    pending = [it for it in all_items if not already_transcribed(it["audio_url"])]

    if not pending:
        print("All episodes already transcribed. Nothing to do.")
        return

    print(f"{len(pending)} episode(s) to process.")

    print(f"Loading Whisper model ({WHISPER_MODEL})...")
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="int8")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    new_files: list[str] = []

    for it in pending:
        title = it["title"]
        url = it["audio_url"]
        published = it["published"]
        base = transcript_basename(title, published, url)

        ext = os.path.splitext(urlparse(url).path)[1].lower() or ".mp3"
        mp3_path = os.path.join(WORKING_DIR, base + ext)

        print(f"\n── {title}")

        # 1. Download
        time.sleep(random.uniform(*BETWEEN_DOWNLOAD_SLEEP) * random.uniform(*JITTER))
        print("  Downloading...")
        try:
            download_episode(session, url, mp3_path)
        except Exception as exc:
            print(f"  SKIP (download error): {exc}")
            continue

        # 2. Transcribe
        print("  Transcribing...")
        try:
            segments, _info = model.transcribe(
                mp3_path,
                beam_size=BEAM_SIZE,
                language=LANGUAGE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": MIN_SILENCE_MS},
                word_timestamps=False,
            )
            segs = list(segments)  # consume generator before closing file
        except Exception as exc:
            print(f"  SKIP (transcription error): {exc}")
            os.remove(mp3_path)
            continue

        # 3. Write outputs
        txt_path = os.path.join(TXT_DIR, base + ".txt")
        srt_path = os.path.join(SRT_DIR, base + ".srt")
        vtt_path = os.path.join(VTT_DIR, base + ".vtt")

        write_txt(txt_path, segs)
        write_srt(srt_path, segs)
        write_vtt(vtt_path, segs)
        new_files.extend([txt_path, srt_path, vtt_path])
        print(f"  ✓ {base}")

        # 4. Remove mp3
        os.remove(mp3_path)

    # Clean up working dir if now empty
    try:
        os.rmdir(WORKING_DIR)
    except OSError:
        pass

    # 5. Git commit
    if new_files:
        print(f"\nCommitting {len(new_files)} file(s)...")
        repo_root = find_repo_root(TRANSCRIPTS_DIR)
        git_commit(new_files, repo_root)
        print("Done.")
    else:
        print("\nNo new transcripts produced.")


if __name__ == "__main__":
    main()
