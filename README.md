# Veda Continuity Builder v2

Public GitHub repo workflow that:
- cleans raw transcript files with Gemma 4 31B
- repairs merged-file boundary mistakes using adjacent episode context
- analyzes continuity with Gemini 3.1 Flash-Lite
- writes cleaned TXT and analysis JSON locally
- batch uploads to Hugging Face

Required secrets:
- HF_TOKEN
- GEMINI_API_KEY

Recommended repo variables:
- HF_SOURCE_REPO_ID
- HF_SOURCE_REPO_TYPE
- HF_SOURCE_FOLDER
- HF_OUTPUT_REPO_ID
- HF_OUTPUT_REPO_TYPE
- HF_OUTPUT_FOLDER
- BATCH_SIZE
- REQUEST_DELAY_SECONDS
- RETRY_BASE_SECONDS
- MAX_RETRIES
- CLEAN_MODEL
- ANALYSIS_MODEL


## Open Short-Drama / Series Rights Research

A new research folder tracks openly licensed historic series candidates, per-file rights checks, dubbing status, and a CPU-only weekly metadata refresher. **Candidate listings are not licence clearance**, and the current evidence does not validate 2,000-3,000 modern vertical short-drama series with verified Hindi/English dubbing and commercial re-publication rights.

- [Research overview](research/open_short_drama_series/README.md)
- [178 series title candidates (CSV)](research/open_short_drama_series/reruncentury_series_candidates.csv)
- [Sources and licence notes](research/open_short_drama_series/SOURCES_AND_LICENSE_NOTES.md)
- [Research report (Markdown)](research/open_short_drama_series/RESEARCH_REPORT.md)
