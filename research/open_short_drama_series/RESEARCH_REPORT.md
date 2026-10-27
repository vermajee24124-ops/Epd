# Research report: free and openly licensed short-drama series

**Date:** 27 October 2026

## Executive result

I could not verify a public, ready-to-publish catalogue of 2,000-3,000 modern vertical short-drama series where the actual full videos are free to download, commercial re-publication rights are clear, and Hindi/English dubbing is verified. This repository therefore publishes a truthful candidate dataset and an automatic metadata refresh workflow rather than inventing or laundering a larger count.

## Sources and counts

- **RerunCentury:** 178 series title records are included in `reruncentury_series_candidates.csv`. The site indexes classic television and links to Internet Archive copies, but individual copies and territories still need review. It is not a catalogue of modern vertical mini-dramas.
- **Public Reel:** its serial index/API reports 309 historic serial records based on automated analysis of U.S. copyright registry records. This is evidence for research, not a blanket legal clearance. Check every chapter and media edition.
- **Wikimedia Commons:** actual videos can be discovered through per-file metadata. The exact licence, attribution and work version must be reviewed for every file.
- **PublicDomainM3U:** a community playlist links to media hosted elsewhere. It does not host files and the playlist is not a licence grant.
- **M-Drama:** 8,102 clip annotations/32,361 QA records are described, but original videos are absent and the dataset is CC BY-NC 4.0, which is not appropriate for a normal commercial app without separate permission.

## Suggested workflow

1. Use the 178 title list for discovery, not auto-publication.
2. Run `.github/workflows/refresh-open-drama-series.yml` to refresh metadata weekly or manually.
3. Inspect exact file-level rights, applicable countries, soundtrack/restoration and poster rights.
4. Verify Hindi/English audio and subtitle tracks independently.
5. Start by publishing a smaller batch (25-50) with documented commercial re-use rights.
6. Grow modern romance, revenge, family and billionaire mini-drama series by obtaining written licences from independent creators or rights holders.

## Rights rule

Do not treat “free to watch” as “free to redistribute.” Do not treat a video page in an open player app, an API response, a metadata licence, or a U.S. registry status as permission to re-host the underlying file. CC BY/CC BY-SA can allow commercial use under conditions; CC BY-NC is non-commercial; CC BY-ND has restrictions for adaptations. Verify exact terms.

## Language and dubbing

Hindi/English dub status is unknown for the candidate titles. Do not label a title as dubbed without checking the actual episode audio or a licensed subtitle/dubbing file. Dubbing a protected work may require separate authorisation.

## Primary links

- https://www.reruncentury.com/ia/
- https://www.reruncentury.com/about/
- https://publicreel.org/collections/serials
- https://publicreel.org/api/collections/serials.json?limit=5000
- https://commons.wikimedia.org/wiki/Category:Film_serials
- https://www.mediawiki.org/wiki/Wikimedia_APIs/Access_policy
- https://github.com/OnlineM3U/publicdomainm3u
- https://huggingface.co/datasets/yixin1121/M-Drama
- https://ask.loc.gov/film-tv-video/faq/302791
- https://creativecommons.org/share-your-work/cclicenses/

This is a discovery research report, not legal advice.
