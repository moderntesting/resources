import os, re, glob, json, argparse
from unidecode import unidecode
from tqdm import tqdm
from faster_whisper import WhisperModel
import pandas as pd

try:
    from mutagen import File as MutagenFile
except Exception:
    MutagenFile = None

# -------- utils --------
AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def slugify(text, max_len=120):
    text = unidecode(text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-")
    return text[:max_len] or "episode"

def format_timestamp(seconds, srt=False):
    if seconds is None: seconds = 0.0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    mm = (ms % 3600000) // 60000
    ss = (ms % 60000) // 1000
    mmm = ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}{',' if srt else '.'}{mmm:03d}"

def write_txt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            text = (s.text or "").strip()
            if text:
                f.write(text + " ")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")

def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        i = 1
        for s in segments:
            text = (s.text or "").strip()
            if not text: continue
            start = format_timestamp(s.start, srt=True)
            end = format_timestamp(s.end, srt=True)
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            i += 1

def write_vtt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segments:
            text = (s.text or "").strip()
            if not text: continue
            start = format_timestamp(s.start)
            end = format_timestamp(s.end)
            f.write(f"{start} --> {end}\n{text}\n\n")

def get_audio_files(directory):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(files)

def extract_tags(filepath):
    """Best-effort tag extraction via mutagen (if available)."""
    meta = {
        "episode_title": os.path.splitext(os.path.basename(filepath))[0],
        "published": "",
        "artist": "",
        "album": "",
        "description": ""
    }
    if MutagenFile is None:
        return meta
    try:
        mf = MutagenFile(filepath, easy=True)
        if not mf:
            return meta

        def first(k):
            v = mf.tags.get(k)
            if isinstance(v, list) and v:
                return v[0]
            if isinstance(v, str):
                return v
            return ""

        # Common keys
        meta["episode_title"] = first("title") or meta["episode_title"]
        meta["artist"] = first("artist")
        meta["album"] = first("album")
        # date variations
        meta["published"] = first("date") or first("originaldate") or first("year")
        # descriptions vary wildly:
        meta["description"] = first("description") or first("comment")
        # m4a special atoms (if not in easy tags)
        if hasattr(mf, "tags") and hasattr(mf.tags, "keys"):
            # Raw tags for richer formats
            raw = mf.tags
            # iTunes-style atoms
            meta["episode_title"] = raw.get("\xa9nam", [meta["episode_title"]])[0] if raw.get("\xa9nam") else meta["episode_title"]
            meta["artist"] = raw.get("\xa9ART", [meta["artist"]])[0] if raw.get("\xa9ART") else meta["artist"]
            meta["album"] = raw.get("\xa9alb", [meta["album"]])[0] if raw.get("\xa9alb") else meta["album"]
            meta["description"] = raw.get("desc", [meta["description"]])[0] if raw.get("desc") else meta["description"]
            meta["published"] = raw.get("\xa9day", [meta["published"]])[0] if raw.get("\xa9day") else meta["published"]
    except Exception:
        pass
    return meta

# -------- main --------
def run(args):
    ensure_dir(args.out_dir)

    files = get_audio_files(args.audio_dir)
    if not files:
        print(f"No audio found in {args.audio_dir}/ (supported: {', '.join(AUDIO_EXTS)})")
        return

    model = WhisperModel(
        args.model,
        device="auto",              # will use Metal on Apple Silicon
        compute_type="int8"     # big speedup with tiny quality hit
)

    rows = []
    combined_md_parts = []

    for fp in tqdm(files, desc="Transcribing"):
        tags = extract_tags(fp)
        title_for_slug = tags["episode_title"] or os.path.splitext(os.path.basename(fp))[0]
        base = slugify(title_for_slug)

        txt_path = os.path.join(args.out_dir, base + ".txt")
        srt_path = os.path.join(args.out_dir, base + ".srt")
        vtt_path = os.path.join(args.out_dir, base + ".vtt")

        need_transcribe = args.force or not (os.path.exists(txt_path) and os.path.exists(srt_path) and os.path.exists(vtt_path))

        duration_sec = ""
        lang = ""
        if need_transcribe:
            segments, info = model.transcribe(
                fp,
                beam_size=args.beam_size,
                language=(args.language or None),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=args.min_silence_ms),
                word_timestamps=False,
            )
            segs = list(segments)
            write_txt(txt_path, segs)
            write_srt(srt_path, segs)
            write_vtt(vtt_path, segs)
            duration_sec = getattr(info, "duration", "")
            lang = getattr(info, "language", "")
        # compute word count
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                word_count = len(re.findall(r"\w+", f.read()))
        except Exception:
            word_count = ""

        # read transcript text for combined md
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception:
            text = ""

        combined_md_parts.append(
f"""## {tags['episode_title'] or os.path.basename(fp)}
**Published:** {tags['published']}
**Artist:** {tags['artist']}
**Album:** {tags['album']}
**File:** {os.path.basename(fp)}

{text}

---
"""
        )

        rows.append({
            "episode_title": tags["episode_title"],
            "published": tags["published"],
            "artist": tags["artist"],
            "album": tags["album"],
            "description": tags["description"],
            "audio_path": fp,
            "txt_path": txt_path,
            "srt_path": srt_path,
            "vtt_path": vtt_path,
            "language": lang,
            "duration_seconds": duration_sec,
            "word_count": word_count,
        })

    # index + combined
    if rows:
        df = pd.DataFrame(rows)
        # Try to sort by publish date if present; otherwise leave as is
        try:
            df["published_sort"] = pd.to_datetime(df["published"], errors="coerce")
            df.sort_values(by=["published_sort", "episode_title"], inplace=True, na_position="last")
            df.drop(columns=["published_sort"], inplace=True)
        except Exception:
            pass

        index_csv = os.path.join(args.out_dir, "index.csv")
        df.to_csv(index_csv, index=False)

        combined_md = os.path.join(args.out_dir, "all_transcripts.md")
        with open(combined_md, "w", encoding="utf-8") as f:
            f.write("# Transcripts\n\n")
            f.write("\n".join(combined_md_parts))

        print(f"\n✅ Done.\nIndex: {index_csv}\nCombined MD: {combined_md}\nTranscripts dir: {args.out_dir}")
    else:
        print("No rows to index.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transcribe local podcast audio and build an index (no RSS).")
    ap.add_argument("--audio-dir", default="episodes", help="Directory with audio files")
    ap.add_argument("--out-dir", default="transcripts", help="Where to write transcripts and index")
    ap.add_argument("--model", default="large-v3", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    ap.add_argument("--language", default=None, help='Force language (e.g., "en")')
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--min-silence-ms", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="Re-transcribe even if artifacts exist")
    args = ap.parse_args()
    run(args)
