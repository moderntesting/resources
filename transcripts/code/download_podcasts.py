import os, time, random, csv, hashlib
import requests, feedparser
from urllib.parse import urlparse
from tqdm import tqdm
from dateutil import parser as dtparser

RSS_URL = "https://anchor.fm/s/45580f58/podcast/rss"         
OUT_DIR = "episodes"
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")

# Tuning knobs
MAX_RETRIES = 8
BASE_SLEEP = 1.0          # base seconds before backoff
MAX_SLEEP = 120.0         # cap backoff
JITTER = (0.25, 1.25)     # random jitter multiplier
BETWEEN_DOWNLOAD_SLEEP = (0.8, 2.2)  #  pause between episodes to avoid rate limiting
TIMEOUT = (10, 60)        # (connect, read) seconds
USER_AGENT = "PodcastBulkDL/1.1 (+https://example.com)"

os.makedirs(OUT_DIR, exist_ok=True)

def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]

def safe_name(s: str) -> str:
    keep = "-_.() "
    return "".join(ch if ch.isalnum() or ch in keep else "_" for ch in s).strip()

def parse_rss(url):
    feed = feedparser.parse(url)
    if feed.bozo:
        raise RuntimeError(f"RSS parse error: {feed.bozo_exception}")
    items = []
    for e in feed.entries:
        title = getattr(e, "title", "Untitled")
        pub = getattr(e, "published", "") or getattr(e, "pubDate", "")
        try:
            pub_iso = dtparser.parse(pub).date().isoformat() if pub else ""
        except Exception:
            pub_iso = pub
        audio = ""
        if getattr(e, "enclosures", None):
            for enc in e.enclosures:
                if enc.get("href"):
                    audio = enc["href"]
                    break
        if not audio:
            for link in getattr(e, "links", []):
                href = link.get("href", "")
                if href.lower().endswith((".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac")):
                    audio = href
                    break
        if audio:
            items.append({"title": title, "published": pub_iso, "audio_url": audio})
    return items

def get_size(session, url):
    try:
        r = session.head(url, allow_redirects=True, timeout=TIMEOUT)
        if r.ok:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except requests.RequestException:
        pass
    return None

def respectful_sleep(base):
    time.sleep(base * random.uniform(*JITTER))

def download_with_resume(session, url, dest):
    tmp = dest + ".part"
    bytes_already = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    total_size = get_size(session, url)

    headers = {"User-Agent": USER_AGENT}
    if bytes_already and total_size and bytes_already < total_size:
        headers["Range"] = f"bytes={bytes_already}-"

    attempt = 0
    with open(tmp, "ab") as f:
        while True:
            try:
                with session.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
                    if r.status_code in (429, 503, 502, 500):
                        # Respect Retry-After if present
                        ra = r.headers.get("Retry-After")
                        if ra:
                            try:
                                sleep_for = min(MAX_SLEEP, float(ra))
                            except Exception:
                                sleep_for = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
                        else:
                            sleep_for = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
                        attempt += 1
                        if attempt > MAX_RETRIES:
                            raise RuntimeError(f"Max retries reached for {url} (status {r.status_code})")
                        print(f"Rate limited / server error ({r.status_code}). Sleeping {sleep_for:.1f}s...")
                        time.sleep(sleep_for)
                        continue

                    if r.status_code == 416:
                        # Range not satisfiable -> probably already complete
                        break

                    r.raise_for_status()

                    # Progress bar setup
                    if total_size:
                        total = total_size - bytes_already if bytes_already else total_size
                    else:
                        total = None

                    desc = f"↓ {os.path.basename(dest)}"
                    with tqdm(total=total, unit="B", unit_scale=True, initial=0, desc=desc) as pbar:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    break  # success
            except requests.RequestException as ex:
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                sleep_for = min(MAX_SLEEP, BASE_SLEEP * (2 ** attempt))
                print(f"Network error: {ex}. Retry {attempt}/{MAX_RETRIES} in {sleep_for:.1f}s")
                time.sleep(sleep_for)

    # Finalize
    if total_size and os.path.getsize(tmp) < total_size:
        raise RuntimeError(f"Incomplete download: {dest}")
    os.replace(tmp, dest)

def read_manifest():
    idx = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx[row["audio_url"]] = row
    return idx

def append_manifest(row):
    write_header = not os.path.exists(MANIFEST)
    with open(MANIFEST, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title","published","audio_url","file"])
        if write_header:
            w.writeheader()
        w.writerow(row)

def main():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    items = parse_rss(RSS_URL)

    # # could filter by date here to get latest episodes. maybe something like
    # AFTER_DATE = dtparser.parse("2025-01-01").date()  # set your cutoff
    # items = [
    #     it for it in items
    #     if it["published"] and dtparser.parse(it["published"]).date() >= AFTER_DATE
    # ]

    manifest = read_manifest()

    for it in items:
        url = it["audio_url"]
        if url in manifest and os.path.exists(manifest[url]["file"]):
            # already done
            continue

        # build filename: YYYY-MM-DD - title - hash.ext
        title = it["title"] or "Untitled"
        date_prefix = f"{it['published']} - " if it["published"] else ""
        ext = os.path.splitext(urlparse(url).path)[1].lower() or ".mp3"
        base = f"{date_prefix}{safe_name(title)} - {short_hash(url)}{ext}"
        dest = os.path.join(OUT_DIR, base)

        # polite inter-episode pause
        respectful_sleep(random.uniform(*BETWEEN_DOWNLOAD_SLEEP))

        print(f"Downloading: {title}")
        download_with_resume(session, url, dest)
        append_manifest({
            "title": title,
            "published": it["published"],
            "audio_url": url,
            "file": dest
        })

    print("\n✅ All available episodes processed politely.")

if __name__ == "__main__":
    main()
