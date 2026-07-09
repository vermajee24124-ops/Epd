#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from openai import OpenAI

HF_REPO_ID = os.getenv("HF_REPO_ID", "Kumarverma11/PocketFM_Audio")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")
HF_SOURCE_FOLDER = os.getenv("HF_SOURCE_FOLDER", "Episode_0001_to_0200_Fast")
HF_OUTPUT_FOLDER = os.getenv("HF_OUTPUT_FOLDER", "Transcripts_Episode_0001_to_0200_Txt")
QWEN_MODELS = [m.strip() for m in os.getenv("QWEN_MODELS", "qwen3.5-omni-plus,qwen3.5-omni,qwen3.5-omni-flash,qwen3.5-omni-plus-realtime").split(",") if m.strip()]
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://YOUR-WORKSPACE-ID.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "2"))
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "true").lower() in {"1", "true", "yes", "y"}
MAX_BASE64_BYTES = int(os.getenv("MAX_BASE64_BYTES", "9500000"))
HF_TOKEN = os.environ["HF_TOKEN"]
QWEN_API_KEY = os.environ["QWEN_API_KEY"]
WORKDIR = Path(os.getenv("WORKDIR", "/tmp/pocketfm_work"))
AUDIO_DIR = WORKDIR / "audio"
TXT_DIR = WORKDIR / "episodes"
TMP_COMPRESS_DIR = WORKDIR / "compressed"
for d in (AUDIO_DIR, TXT_DIR, TMP_COMPRESS_DIR): d.mkdir(parents=True, exist_ok=True)
PROMPT = os.getenv("TRANSCRIBE_PROMPT", "इस हिंदी ऑडियो ड्रामा को अत्यंत सटीक रूप से शब्दशः ट्रांसक्राइब करो। केवल साफ transcript text दो। JSON, Markdown, summary, explanation, continuity notes, scene notes या नई कहानी मत जोड़ो। Narration, dialogue, character names और घटनाओं का क्रम जस का तस रखो। यदि audio में English/Hinglish शब्द बोले गए हों तो वैसे ही लिखो। Output केवल transcript होना चाहिए.")
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma"}


def ep_no(path: str) -> Optional[int]:
    m = re.search(r"Episode_(\d{4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def source_rows() -> List[Tuple[int, str]]:
    hf = HfApi(token=HF_TOKEN)
    rows = []
    for item in hf.list_repo_tree(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, recursive=True, expand=False):
        path = getattr(item, "path", "")
        if not path.startswith(HF_SOURCE_FOLDER + "/"):
            continue
        if Path(path).suffix.lower() not in AUDIO_EXTS:
            continue
        ep = ep_no(path)
        if ep and 1 <= ep <= 200:
            rows.append((ep, path))
    rows.sort()
    nums = [e for e, _ in rows]
    if nums != list(range(1, 201)):
        missing = sorted(set(range(1, 201)) - set(nums))
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        raise RuntimeError(f"Source sequence invalid. Missing={missing}, duplicates={dupes}")
    return rows


def remote_done() -> set[int]:
    hf = HfApi(token=HF_TOKEN)
    done = set()
    for item in hf.list_repo_tree(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, recursive=True, expand=False):
        path = getattr(item, "path", "")
        if not path.startswith(HF_OUTPUT_FOLDER + "/"):
            continue
        if Path(path).suffix.lower() != ".txt":
            continue
        ep = ep_no(path)
        if ep:
            done.add(ep)
    return done


def download_audio(repo_path: str) -> Path:
    return Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=repo_path, token=HF_TOKEN, local_dir=str(AUDIO_DIR)))


def compress_if_needed(p: Path, ep: int) -> Path:
    est_b64 = ((p.stat().st_size + 2) // 3) * 4
    if est_b64 <= MAX_BASE64_BYTES:
        return p
    out = TMP_COMPRESS_DIR / f"Episode_{ep:04d}_api.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", str(p), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(out)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ((out.stat().st_size + 2) // 3) * 4 > MAX_BASE64_BYTES:
        raise RuntimeError(f"Episode {ep:04d} is still too large after compression.")
    return out


def build_client() -> OpenAI:
    return OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, timeout=1200)


def transcribe_once(client: OpenAI, model: str, audio_path: Path, ep: int) -> str:
    api_audio = compress_if_needed(audio_path, ep)
    audio_b64 = base64.b64encode(api_audio.read_bytes()).decode("utf-8")
    fmt = api_audio.suffix.lower().lstrip(".") or "mp3"
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": f"data:;base64,{audio_b64}", "format": fmt}}, {"type": "text", "text": f"Episode {ep}\n\n{PROMPT}"}]}],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": ENABLE_THINKING},
    )
    parts = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    txt = "".join(parts).strip()
    if not txt:
        raise RuntimeError("Empty transcript received.")
    return txt


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def upload_batch(paths: List[Path]) -> None:
    hf = HfApi(token=HF_TOKEN)
    eps = [ep_no(p.name) for p in paths if ep_no(p.name) is not None]
    ops = [CommitOperationAdd(path_in_repo=f"{HF_OUTPUT_FOLDER}/{p.name}", path_or_fileobj=str(p)) for p in paths]
    start_ep, end_ep = min(eps), max(eps)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hf.create_commit(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, operations=ops, commit_message=f"Add TXT transcripts Episode {start_ep:04d}-{end_ep:04d}", token=HF_TOKEN)
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))


def main() -> None:
    rows = source_rows()
    done = remote_done()
    print(f"Source verified: {len(rows)} episodes")
    print(f"Already on Hugging Face: {len(done)}/200")
    client = build_client()
    batch: List[Path] = []
    for idx, (ep, repo_path) in enumerate(rows, start=1):
        if ep in done:
            print(f"[{idx:03d}/200] Episode {ep:04d} already exists. SKIP")
            continue
        txt = TXT_DIR / f"Episode_{ep:04d}.txt"
        audio = download_audio(repo_path)
        transcript = None
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                for model in QWEN_MODELS:
                    try:
                        transcript = transcribe_once(client, model, audio, ep)
                        break
                    except Exception as model_exc:
                        msg = str(model_exc).lower()
                        if "429" in msg or "rate limit" in msg or "limit" in msg:
                            continue
                        continue
                if transcript is not None:
                    break
            except Exception as exc:
                last_err = exc
            if transcript is None:
                time.sleep(RETRY_BASE_SECONDS * (2 ** (attempt - 1)))
        if transcript is None:
            raise RuntimeError(f"Episode {ep:04d} failed: {last_err}")
        save_text(txt, transcript)
        batch.append(txt)
        try:
            audio.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"[{idx:03d}/200] Saved {txt.name}")
        if len(batch) >= BATCH_SIZE:
            upload_batch(batch)
            for p in batch: p.unlink(missing_ok=True)
            batch.clear()
        time.sleep(REQUEST_DELAY_SECONDS)
    if batch:
        upload_batch(batch)
        for p in batch: p.unlink(missing_ok=True)
        batch.clear()
    missing = sorted(set(range(1, 201)) - remote_done())
    if missing:
        raise RuntimeError(f"Final verification failed. Missing: {missing}")
    print("FINAL SUCCESS")


if __name__ == "__main__":
    main()
