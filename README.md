# PocketFM Qwen Pipeline

This repository contains a GitHub Actions pipeline that:

1. Downloads source audio from a Hugging Face dataset folder.
2. Sends each audio file to your Qwen API.
3. Saves each transcript locally in a GitHub repo folder named `episodes/`.
4. Commits that folder back to GitHub in one commit.
5. Uploads the completed transcript folder to Hugging Face in one batch commit.

## Files

- `main.py`
- `.github/workflows/transcribe.yml`
- `requirements.txt`

## Secrets to add

- `HF_TOKEN`
- `HF_SOURCE_REPO_ID`
- `HF_SOURCE_REPO_TYPE`
- `HF_SOURCE_FOLDER`
- `HF_OUTPUT_REPO_ID`
- `HF_OUTPUT_REPO_TYPE`
- `HF_OUTPUT_FOLDER`
- `QWEN_API_URL`
- `QWEN_API_KEY`
- `QWEN_MODEL`
- `REQUEST_DELAY_SECONDS`
- `RETRY_BASE_SECONDS`
- `MAX_RETRIES`
- `TRANSCRIBE_PROMPT`

## Important

GitHub-hosted runners execute workflow jobs in ephemeral, clean isolated virtual machines, so anything you want to keep must be written back to a repository or uploaded somewhere during the job. GitHub Actions workflows are defined by YAML files, and the `huggingface_hub` library supports both downloading files and uploading folders/files to the Hub. citeturn932264search4turn494328search1turn494328search2turn494328search4turn494328search5

If your Qwen provider uses a request schema different from the one in `call_qwen_transcriber()`, edit only that function.
