#!/usr/bin/env python3
"""Collect candidate series/media metadata only; never certify copyright clearance."""
from __future__ import annotations
import argparse, csv, json, re, sys, time
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

UA = "OpenDramaRightsResearch/1.0 (metadata discovery; descriptive User-Agent)"
PUBLIC_REEL = "https://publicreel.org/api/collections/serials.json?limit=5000"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
PLAYLIST = "https://raw.githubusercontent.com/OnlineM3U/publicdomainm3u/main/series.m3u"

def fetch(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
    with urlopen(req, timeout=timeout) as response:
        return response.read()

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def collect_public_reel() -> list[dict]:
    data = json.loads(fetch(PUBLIC_REEL).decode("utf-8"))
    rows = []
    for item in data.get("results", []):
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        source = item.get("url") or "/collections/serials"
        rows.append({
            "series_title": title,
            "year": item.get("year", ""),
            "chapter_count": len(item.get("chapters") or []),
            "registry_status": item.get("status_label", item.get("status", "unknown")),
            "registry_evidence": item.get("evidence", ""),
            "chapter_status_summary": item.get("chapters_summary", ""),
            "watch_url": item.get("watch_url") or "",
            "source_url": urljoin("https://publicreel.org/", source),
            "language_status": "Not verified; dub tracks not checked",
            "rights_gate": "REVIEW_REQUIRED - US registry analysis is not global legal clearance; check every chapter/version",
            "commercial_reuse": "Not approved automatically",
        })
    return rows

def strip_html(value: object) -> str:
    return re.sub(r"<[^>]+>", "", str(value or "")).strip()

def collect_commons_video_candidates() -> list[dict]:
    rows, cont = [], {}
    fields = ["file_title", "file_page_url", "direct_file_url", "mime", "licence_short_name",
              "licence_url", "artist_credit", "attribution_text", "usage_terms", "screening_status",
              "language_dub_status"]
    for _ in range(10):
        params = {
            "action": "query", "format": "json", "generator": "categorymembers",
            "gcmtitle": "Category:Film_serials", "gcmtype": "file", "gcmlimit": "500",
            "prop": "imageinfo", "iiprop": "url|mime|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|Credit|UsageTerms|Attribution",
        }
        params.update(cont)
        data = json.loads(fetch(COMMONS_API + "?" + urlencode(params)).decode("utf-8"))
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            title = page.get("title", "")
            ii = (page.get("imageinfo") or [{}])[0]
            mime = ii.get("mime", "")
            direct = ii.get("url", "")
            if not (mime.startswith("video/") or re.search(r"\.(webm|ogv|mp4|mov|mkv)(?:\?|$)", direct, re.I)):
                continue
            ext = ii.get("extmetadata", {}) or {}
            def val(key: str) -> str:
                return strip_html((ext.get(key) or {}).get("value", ""))
            licence, licence_url = val("LicenseShortName"), val("LicenseUrl")
            maybe_ok = bool(re.search(r"CC0|CC BY(?:-SA)?|public domain", licence, re.I))
            rows.append({
                "file_title": title,
                "file_page_url": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                "direct_file_url": direct,
                "mime": mime,
                "licence_short_name": licence,
                "licence_url": licence_url,
                "artist_credit": val("Artist") or val("Credit"),
                "attribution_text": val("Attribution"),
                "usage_terms": val("UsageTerms"),
                "screening_status": "LICENCE_CANDIDATE - manual rights/territory check required" if maybe_ok else "REVIEW_REQUIRED - licence not confidently recognised",
                "language_dub_status": "Not checked",
            })
        cont = data.get("continue", {})
        if not cont:
            break
        time.sleep(0.3)
    return rows

def collect_playlist_entries() -> list[dict]:
    text = fetch(PLAYLIST).decode("utf-8", "replace")
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXTINF:"):
            group = re.search(r'group-title="([^"]+)"', line)
            title = line.split(",", 1)[1].strip() if "," in line else ""
            rows.append({"group_title": group.group(1) if group else "", "episode_title": title,
                         "video_url": "", "source_list": PLAYLIST,
                         "rights_status": "Candidate link only; playlist does not independently prove rights for exact item/version"})
        elif not line.startswith("#") and rows and not rows[-1]["video_url"]:
            rows[-1]["video_url"] = line
    return rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="generated")
    args = parser.parse_args()
    out, errors = Path(args.out_dir), []
    out.mkdir(parents=True, exist_ok=True)
    specs = [
        ("Public Reel", "publicreel_serial_candidates.csv", collect_public_reel,
         ["series_title","year","chapter_count","registry_status","registry_evidence","chapter_status_summary","watch_url","source_url","language_status","rights_gate","commercial_reuse"]),
        ("Wikimedia Commons", "commons_film_serial_video_candidates.csv", collect_commons_video_candidates,
         ["file_title","file_page_url","direct_file_url","mime","licence_short_name","licence_url","artist_credit","attribution_text","usage_terms","screening_status","language_dub_status"]),
        ("PublicDomainM3U", "publicdomainm3u_episode_links.csv", collect_playlist_entries,
         ["group_title","episode_title","video_url","source_list","rights_status"]),
    ]
    for label, filename, function, columns in specs:
        try:
            rows = function()
            write_csv(out / filename, rows, columns)
            print(f"{label}: wrote {len(rows)} candidate rows to {filename}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            print(errors[-1], file=sys.stderr)
    status = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_errors": errors,
        "warning": "Metadata discovery only. No row is automatically cleared for commercial use, rehosting, translation or dubbing."
    }
    (out / "refresh_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 1 if len(errors) == len(specs) else 0

if __name__ == "__main__":
    raise SystemExit(main())
