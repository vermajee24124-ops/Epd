#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from huggingface_hub import HfApi, hf_hub_download

LOG = logging.getLogger("pocketfm")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma"}
OUTPUT_TEXT_EXT = ".json"
MANIFEST_NAME = "manifest.json"


def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value or ""


def parse_range(filename: str) -> Optional[Tuple[int, int]]:
    import re
    s = Path(filename).stem.lower()

    m = re.search(r"episode[_\s-]*(\d+)[_\s-]*(?:to|-)[_\s-]*(\d+)", s)
    if not m:
        m = re.search(r"episode[_\s-]*(\d+)[_\s-]+(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = re.search(r"episode[_\s-]*(\d+)", s)
    if m:
        n = int(m.group(1))
        return n, n

    return None


def list_remote_audio_paths(api: HfApi, repo_id: str, repo_type: str, prefix: str) -> List[str]:
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        recursive=True,
        expand=False,
    )

    paths: List[str] = []
    prefix = prefix.strip("/")

    for item in items:
        path = getattr(item, "path", "")
        if Path(path).suffix.lower() not in AUDIO_EXTS:
            continue
        if prefix and not path.startswith(prefix + "/"):
            continue
        paths.append(path)

    return sorted(paths)


def remote_file_exists(api: HfApi, repo_id: str, repo_type: str, path_in_repo: str) -> bool:
    try:
        tree = api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            recursive=True,
            expand=False,
        )
        target = path_in_repo.replace("\\", "/")
        return any(getattr(item, "path", "").replace("\\", "/") == target for item in tree)
    except Exception:
        return False


def download_audio(local_dir: Path, repo_id: str, repo_type: str, repo_path: str, token: str) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=repo_path,
        token=token,
        local_dir=str(local_dir),
    )
    return Path(local_path)


def call_qwen_transcriber(
    *,
    api_url: str,
    api_key: str,
    model: str,
    audio_path: Path,
    episode_no: int,
    prompt: str,
    timeout: int = 300,
) -> Dict:
    """
    Generic OpenAI-compatible multimodal request.
    If your free Qwen endpoint uses a different schema, edit only this function.
    """
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    audio_format = audio_path.suffix.lower().lstrip(".") or "mp3"

    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You transcribe long-form audio drama. "
                    "Return strict JSON only, no markdown."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Episode number: {episode_no}\n"
                            f"Task: {prompt}\n\n"
                            "Return JSON with these keys:\n"
                            "- transcript\n"
                            "- title\n"
                            "- summary\n"
                            "- continuity_notes\n"
                            "- scene_notes\n"
                            "- confidence\n"
                            "- timestamp_notes\n"
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": audio_format,
                        },
                    },
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    text = None
    if isinstance(data, dict):
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    text = msg.get("content")
                else:
                    text = choice.get("text")
        elif "output_text" in data:
            text = data.get("output_text")
        elif "content" in data:
            text = data.get("content")

    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)

    if not text:
        raise RuntimeError(f"Could not read text from response: {data}")

    return json.loads(text.strip())


def upload_text_file(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    token: str,
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
    )


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    hf_token = env("HF_TOKEN", required=True)
    source_repo = env("HF_SOURCE_REPO_ID", required=True)
    source_repo_type = env("HF_SOURCE_REPO_TYPE", "dataset")
    source_folder = env("HF_SOURCE_FOLDER", required=True)

    output_repo = env("HF_OUTPUT_REPO_ID", source_repo)
    output_repo_type = env("HF_OUTPUT_REPO_TYPE", "dataset")
    output_folder = env("HF_OUTPUT_FOLDER", "Transcripts_Episode_0001_0200")

    qwen_api_url = env("QWEN_API_URL", required=True)
    qwen_api_key = env("QWEN_API_KEY", required=True)
    qwen_model = env("QWEN_MODEL", "qwen-3.5-omni-plus")

    request_delay_seconds = float(env("REQUEST_DELAY_SECONDS", "2.0"))
    retry_base_seconds = float(env("RETRY_BASE_SECONDS", "8.0"))
    max_retries = int(env("MAX_RETRIES", "5"))

    prompt = env(
        "TRANSCRIBE_PROMPT",
        (
            "Transcribe the audio exactly. "
            "Keep ordering, speaker turns, narration, and continuity notes. "
            "Return strict JSON."
        ),
    )

    # This folder is local inside the GitHub Actions workspace.
    local_repo_folder = Path(env("LOCAL_EPISODE_FOLDER", "episodes"))
    local_repo_folder.mkdir(parents=True, exist_ok=True)

    workdir = Path(env("WORKDIR", "/tmp/pocketfm_qwen_work"))
    local_audio_dir = workdir / "audio"
    state_dir = workdir / "state"
    local_audio_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    done_path = state_dir / "done.json"
    manifest_path = state_dir / MANIFEST_NAME

    done = {}
    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))

    api = HfApi(token=hf_token)

    audio_paths = list_remote_audio_paths(
        api=api,
        repo_id=source_repo,
        repo_type=source_repo_type,
        prefix=source_folder,
    )
    if not audio_paths:
        raise RuntimeError(f"No audio files found under {source_folder} in {source_repo}")

    records = []

    for idx, repo_path in enumerate(audio_paths, start=1):
        filename = Path(repo_path).name
        rr = parse_range(filename)
        episode_no = rr[0] if rr else idx

        out_name = f"Episode_{episode_no:04d}.json"
        local_out = local_repo_folder / out_name
        remote_out = f"{output_folder}/{out_name}"

        if done.get(out_name) == "uploaded":
            records.append(
                {
                    "episode": episode_no,
                    "source": repo_path,
                    "output": out_name,
                    "status": "skipped",
                }
            )
            continue

        if remote_file_exists(api, output_repo, output_repo_type, remote_out):
            done[out_name] = "uploaded"
            save_json(done_path, done)
            records.append(
                {
                    "episode": episode_no,
                    "source": repo_path,
                    "output": out_name,
                    "status": "already_remote",
                }
            )
            continue

        LOG.info("[%d/%d] downloading %s", idx, len(audio_paths), repo_path)
        local_audio = download_audio(
            local_dir=local_audio_dir,
            repo_id=source_repo,
            repo_type=source_repo_type,
            repo_path=repo_path,
            token=hf_token,
        )

        success = False
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                LOG.info("[%d/%d] transcribing episode %s attempt %d", idx, len(audio_paths), episode_no, attempt)
                result = call_qwen_transcriber(
                    api_url=qwen_api_url,
                    api_key=qwen_api_key,
                    model=qwen_model,
                    audio_path=local_audio,
                    episode_no=episode_no,
                    prompt=prompt,
                )
                result.setdefault("episode", episode_no)
                result.setdefault("source_repo_path", repo_path)
                result.setdefault("source_file", filename)
                result.setdefault("model", qwen_model)
                result.setdefault("status", "ok")

                save_json(local_out, result)

                records.append(
                    {
                        "episode": episode_no,
                        "source": repo_path,
                        "output": out_name,
                        "status": "created_local",
                    }
                )

                # Immediate upload to the Hugging Face output folder, but only after the
                # transcript file has been written locally.
                upload_text_file(
                    api=api,
                    repo_id=output_repo,
                    repo_type=output_repo_type,
                    token=hf_token,
                    local_path=local_out,
                    path_in_repo=remote_out,
                    commit_message=f"Add transcript Episode {episode_no:04d}",
                )

                done[out_name] = "uploaded"
                save_json(done_path, done)
                success = True
                break

            except Exception as exc:
                last_error = str(exc)
                LOG.warning("Episode %s failed attempt %d: %s", episode_no, attempt, exc)
                if attempt < max_retries:
                    wait = retry_base_seconds * (2 ** (attempt - 1))
                    time.sleep(wait)

        if not success:
            records.append(
                {
                    "episode": episode_no,
                    "source": repo_path,
                    "output": out_name,
                    "status": "failed",
                    "error": last_error,
                }
            )
            save_json(state_dir / "failed_last.json", {"episode": episode_no, "error": last_error})
            raise RuntimeError(f"Episode {episode_no} failed after retries: {last_error}")

        time.sleep(request_delay_seconds)

    save_json(
        manifest_path,
        {
            "source_repo": source_repo,
            "source_folder": source_folder,
            "output_repo": output_repo,
            "output_folder": output_folder,
            "audio_file_count": len(audio_paths),
            "records": records,
        },
    )
    LOG.info("Done. Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
