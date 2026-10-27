# Open Short-Drama / Series Rights Research

**Research snapshot:** 27 October 2026  
**Location in repository:** `research/open_short_drama_series/`  
**Purpose:** Build a license-aware catalogue for a future free-to-watch short-drama app. Discovery data is not legal advice and is not a content licence.

## What this folder contains

- `reruncentury_series_candidates.csv`: 178 classic TV series titles visible in the reviewed RerunCentury index. They are candidate records only, not app-approved media.
- `sources.csv`: source list, what each source actually provides, and reuse limits.
- `SOURCES_AND_LICENSE_NOTES.md`: practical rights review, language/dubbing and publication rules.
- `scripts/refresh_catalog.py`: CPU-only metadata collector for Public Reel serial records, Wikimedia Commons Film Serial category videos and the PublicDomainM3U series playlist.
- `generated/`: refreshed CSV snapshots produced by the workflow after it runs.

## Important finding

This research has **not** verified a ready-made library of 2,000-3,000 modern vertical short-drama series with downloadable full videos, clear commercial re-publication rights, and confirmed Hindi/English dubbing. The 178 rows below are historic US television title candidates. Public Reel identifies 309 historic serial records from US registration/renewal evidence; that is not blanket legal clearance. Do not inflate counts by counting episodes as separate series, copying copyrighted app streams or treating open-source player code as a video licence.

## Refresh automatically

GitHub Actions workflow: [Refresh Open Drama Series Research](../../../.github/workflows/refresh-open-drama-series.yml)

- It can be started manually from the repository's **Actions** tab.
- It is scheduled weekly, but scheduled Actions can be delayed or disabled by repository settings.
- It runs on a normal CPU runner; no GPU is needed for metadata discovery.
- Its outputs are candidates. Nothing is marked fully cleared for publication by automation.

## Publication gate

Before any media is put in the app, manually confirm the exact episode/video file, source and licence, India plus target-country status, commercial streaming and storage rights, soundtrack/restoration/subtitle/poster rights, attribution text, and whether Hindi/English dubs or adaptations are authorised. Keep unknowns as unknown; don't infer dubbing from a title.

**Best initial route:** use candidate metadata to discover titles, then publish only exact files whose rights are established. Modern short dramas will usually require direct permission from rights holders or creator-provided licences.
