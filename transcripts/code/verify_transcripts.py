#!/usr/bin/env python3
"""
verify_transcripts.py

Check that every episode in the RSS feed has a complete, plausible transcript set:
  - exactly one .txt, .srt and .vtt file (matched by the audio URL hash)
  - no file is nearly empty
  - the .txt has at least MIN_WPM words per minute of episode duration
  - the last .srt cue lands near the end of the episode

Exits 0 when everything passes, 1 otherwise. Run with the pipeline's venv:
  ../venv/bin/python3 verify_transcripts.py
"""

from __future__ import annotations

import glob
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import feedparser
import requests

import podcast_pipeline as p

MIN_BYTES = 200
MIN_WPM = 80
MIN_COVERAGE = 0.85      # last srt cue must reach this fraction of the duration...
MAX_GAP_SECONDS = 120    # ...unless it's within this many seconds of the end


def duration_seconds(value) -> float | None:
    if not value:
        return None
    total = 0.0
    for part in str(value).split(":"):
        total = total * 60 + float(part)
    return total


def last_cue_seconds(path: str) -> float:
    with open(path, encoding="utf-8") as f:
        times = re.findall(r"--> (\d+):(\d+):(\d+)[.,](\d+)", f.read())
    if not times:
        return 0.0
    h, m, s, ms = map(int, times[-1])
    return h * 3600 + m * 60 + s + ms / 1000


def check_episode(entry) -> list[str]:
    url = next((x["href"] for x in entry.get("enclosures", []) if x.get("href")), None)
    if not url:
        return ["no audio enclosure"]

    h = p.short_hash(url)
    files = {ext: glob.glob(os.path.join(d, f"*{h}{ext}")) for d, ext in p.OUTPUTS}
    issues = []
    for ext, hits in files.items():
        if not hits:
            issues.append(f"missing {ext}")
        elif len(hits) > 1:
            issues.append(f"{len(hits)} {ext} files")
        elif os.path.getsize(hits[0]) < MIN_BYTES:
            issues.append(f"tiny {ext} ({os.path.getsize(hits[0])}B)")

    dur = duration_seconds(entry.get("itunes_duration"))
    if not dur:
        return issues
    if len(files[".txt"]) == 1:
        with open(files[".txt"][0], encoding="utf-8") as f:
            words = len(f.read().split())
        wpm = words / (dur / 60)
        if wpm < MIN_WPM:
            issues.append(f"low word count: {words} words for {dur/60:.0f} min ({wpm:.0f} wpm)")
    if len(files[".srt"]) == 1:
        end = last_cue_seconds(files[".srt"][0])
        if end < dur * MIN_COVERAGE and dur - end > MAX_GAP_SECONDS:
            issues.append(f"srt ends at {end/60:.1f} of {dur/60:.1f} min")
    return issues


def main() -> int:
    raw = requests.get(p.RSS_URL, headers={"User-Agent": p.USER_AGENT}, timeout=p.TIMEOUT).content
    feed = feedparser.parse(raw)
    if not feed.entries:
        print("RSS feed returned no entries")
        return 1

    problems = [(e.get("title", "Untitled"), check_episode(e)) for e in feed.entries]
    problems = [(t, i) for t, i in problems if i]

    ok = len(feed.entries) - len(problems)
    print(f"{len(feed.entries)} feed episodes, {ok} OK, {len(problems)} with problems")
    for title, issues in problems:
        print(f"  {title}: {'; '.join(issues)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
