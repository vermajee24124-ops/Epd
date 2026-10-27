# Source and licence review notes

Research snapshot: 27 October 2026.

## What the sources actually prove

| Source | Data found | What it does not prove |
|---|---|---|
| [RerunCentury](https://www.reruncentury.com/ia/) | 178 historic TV show titles in the reviewed index; the site describes 170+ series and around 1,300 episodes | It is not a blanket licence for each copy, soundtrack, restoration or territory. |
| [Public Reel serials](https://publicreel.org/collections/serials) | 309 historic chapter-serial records based on automated U.S. copyright registration/renewal analysis | Not global copyright clearance. The site says each chapter must be reviewed separately. |
| [Wikimedia Commons film serials](https://commons.wikimedia.org/wiki/Category:Film_serials) | Some actual media files with file-page license and artist metadata | Category membership alone does not clear the file. Review each file and rights in each distribution territory. |
| [PublicDomainM3U](https://github.com/OnlineM3U/publicdomainm3u) | Curated links to publicly accessible historical media; repository says it does not host media | A link or playlist is not independent proof that a copy is legally redistributable. This is not a large modern short-drama dataset. |
| [M-Drama](https://huggingface.co/datasets/yixin1121/M-Drama) | 8,102 clip annotations and 32,361 QA records; source videos are not included | CC BY-NC 4.0 is not suitable for a normally monetised app without separate permission; it is not a video library. |
| [U.S. Library of Congress FAQ](https://ask.loc.gov/film-tv-video/faq/302791) | Copyright research guidance | U.S. status is not an automatic answer for India or other territories. |

## Licence handling

- **CC0 / Public Domain:** usually the simplest candidates, but verify the exact edition and the countries where the app will operate.
- **CC BY:** commercial reuse is generally allowed under licence conditions; preserve attribution, licence link and modification notes.
- **CC BY-SA:** commercial use can be allowed, but ShareAlike obligations can apply to adaptations.
- **CC BY-NC:** exclude from a normal monetised streaming app unless separate permission is obtained.
- **CC BY-ND:** do not distribute modified/adapted versions under that licence; dubbing/translation may raise adaptation issues.
- **Unknown / stream-only / third-party API:** do not publish until the rights holder gives permission.

A credit line is not a substitute for permission. App/player source code may be open-source while its video streams, posters, music and dubbing remain copyrighted. Data metadata and actual media files have separate licences.

## Dubbing/language

No verified Hindi or English dubbing catalogue was found in this research. The 178-title list is dominated by historic U.S. television. Language/dub is deliberately marked not verified. Verify each audio track or subtitle file separately. Creating a new dub may require rights to adapt the underlying work and rights for new music, narration, translations and performances.

## Automation design

The collector in `scripts/refresh_catalog.py` uses only Python's standard library and runs on CPU. It saves candidate metadata from three discovery sources and writes an error/status JSON. It does not scrape paid drama apps, bypass logins, remove DRM, or automatically certify copyright status. Each entry remains `CANDIDATE`/review-required until humans check the exact media file.
