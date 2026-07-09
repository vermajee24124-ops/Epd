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
