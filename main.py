#!/usr/bin/env python3
import base64
import os
import re
import subprocess
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from openai import OpenAI

# ---------- FIXED PROJECT SETTINGS ----------
HF_REPO_ID = "Kumarverma11/PocketFM_Audio"
HF_REPO_TYPE = "dataset"
HF_SOURCE_FOLDER = "Episode_0001_to_0200_Fast"
HF_OUTPUT_FOLDER = "Transcripts_Episode_0001_to_0200"

MODEL = "qwen3.5-omni-plus"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

BATCH_SIZE = 20
MAX_RETRIES = 6
RETRY_BASE_SECONDS = 15
REQUEST_DELAY_SECONDS = 3

LOCAL_EPISODES = Path("/tmp/episodes")
LOCAL_AUDIO = Path("/tmp/audio")
LOCAL_EPISODES.mkdir(parents=True, exist_ok=True)
LOCAL_AUDIO.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.environ["HF_TOKEN"]
QWEN_API_KEY = os.environ["QWEN_API_KEY"]

hf = HfApi(token=HF_TOKEN)
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    timeout=1200,
)

PROMPT = """इस हिंदी ऑडियो ड्रामा को अत्यंत सटीक रूप से शब्दशः ट्रांसक्राइब करो।

आउटपुट में केवल साफ transcript text दो। JSON, Markdown, summary, explanation,
continuity notes, scene notes या अपनी ओर से कोई नई कहानी मत जोड़ो।

Narration, dialogue, character names, घटनाओं का क्रम और episode ending को उसी
क्रम में रखो जिस क्रम में audio में बोला गया है। जहाँ English या Hinglish वास्तव
में बोली गई हो, उसे स्वाभाविक रूप में लिखो। Audio में जो नहीं बोला गया है, उसे मत
लिखो। पूरा transcript दो, बीच का हिस्सा मत छोड़ो।"""


def episode_number(path: str):
    m = re.search(r"Episode_(\d{4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def list_source_episodes():
    rows = []
    for item in hf.list_repo_tree(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo=HF_SOURCE_FOLDER,
        recursive=False,
        expand=False,
    ):
        path = getattr(item, "path", "")
        ep = episode_number(path)
        if ep and 1 <= ep <= 200:
            rows.append((ep, path))

    rows.sort()
    nums = [ep for ep, _ in rows]
    if nums != list(range(1, 201)):
        missing = sorted(set(range(1, 201)) - set(nums))
        duplicates = sorted({n for n in nums if nums.count(n) > 1})
        raise RuntimeError(
            f"Source sequence invalid. Missing={missing}, duplicates={duplicates}"
        )
    return rows


def list_completed_remote():
    completed = set()
    try:
        for item in hf.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            path_in_repo=HF_OUTPUT_FOLDER,
            recursive=False,
            expand=False,
        ):
            ep = episode_number(getattr(item, "path", ""))
            if ep:
                completed.add(ep)
    except Exception:
        pass
    return completed


def prepare_audio_for_base64(source: Path, ep: int) -> Path:
    """
    Qwen docs limit Base64 strings to <10 MB.
    If the original Base64 would be too large, create a speech-friendly
    32 kbps mono MP3 copy locally. The source on Hugging Face is untouched.
    """
    raw_size = source.stat().st_size
    estimated_b64 = ((raw_size + 2) // 3) * 4

    if estimated_b64 < 9_500_000:
        return source

    compressed = LOCAL_AUDIO / f"Episode_{ep:04d}_api.mp3"
    print(
        f"Episode {ep:04d}: Base64 would be too large. "
        "Creating temporary 32 kbps mono MP3."
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000",
            "-b:a", "32k",
            str(compressed),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    encoded_size = ((compressed.stat().st_size + 2) // 3) * 4
    if encoded_size >= 9_500_000:
        raise RuntimeError(
            f"Episode {ep:04d} is still too large for Base64 after compression."
        )
    return compressed


def transcribe(audio_path: Path, ep: int) -> str:
    audio_path = prepare_audio_for_base64(audio_path, ep)
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    fmt = audio_path.suffix.lower().lstrip(".")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio_b64}",
                            "format": fmt,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Episode {ep}\n\n{PROMPT}",
                    },
                ],
            }
        ],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)

    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("Qwen returned an empty transcript.")
    return text


def upload_batch(batch_paths):
    operations = []
    episode_ids = []

    for local_path in batch_paths:
        ep = episode_number(local_path.name)
        episode_ids.append(ep)
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{HF_OUTPUT_FOLDER}/{local_path.name}",
                path_or_fileobj=str(local_path),
            )
        )

    start_ep = min(episode_ids)
    end_ep = max(episode_ids)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"Uploading batch Episode {start_ep:04d}-{end_ep:04d} "
                f"to Hugging Face in ONE commit..."
            )
            hf.create_commit(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                operations=operations,
                commit_message=(
                    f"Add TXT transcripts Episode "
                    f"{start_ep:04d}-{end_ep:04d}"
                ),
                token=HF_TOKEN,
            )
            print("BATCH UPLOAD SUCCESS")
            return
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"HF batch upload failed: {exc}")
            print(f"Retrying after {wait}s...")
            time.sleep(wait)


def main():
    source_rows = list_source_episodes()
    completed = list_completed_remote()

    print("PASS: Exact Episode 0001-0200 source sequence found.")
    print(f"Already safe on private Hugging Face: {len(completed)}/200")

    pending_batch = []

    for position, (ep, repo_path) in enumerate(source_rows, 1):
        if ep in completed:
            print(f"[{position}/200] Episode {ep:04d} already remote. SKIP")
            continue

        print(f"[{position}/200] Downloading Episode {ep:04d}")
        downloaded = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=repo_path,
                token=HF_TOKEN,
                local_dir=str(LOCAL_AUDIO),
            )
        )

        output = LOCAL_EPISODES / f"Episode_{ep:04d}.txt"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(
                    f"[{position}/200] Qwen transcription "
                    f"attempt {attempt}/{MAX_RETRIES}"
                )
                transcript = transcribe(downloaded, ep)
                output.write_text(transcript + "\n", encoding="utf-8")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    break
                wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                print(f"Qwen error: {exc}")
                print(f"Retrying after {wait}s...")
                time.sleep(wait)

        if last_error is not None:
            raise RuntimeError(
                f"Episode {ep:04d} failed after retries: {last_error}"
            )

        pending_batch.append(output)
        print(
            f"[{position}/200] Episode {ep:04d} ready locally. "
            f"Batch={len(pending_batch)}/{BATCH_SIZE}"
        )

        try:
            downloaded.unlink(missing_ok=True)
        except Exception:
            pass

        if len(pending_batch) >= BATCH_SIZE:
            upload_batch(pending_batch)
            completed.update(
                episode_number(p.name) for p in pending_batch
            )
            for p in pending_batch:
                p.unlink(missing_ok=True)
            pending_batch.clear()

        time.sleep(REQUEST_DELAY_SECONDS)

    if pending_batch:
        upload_batch(pending_batch)
        completed.update(episode_number(p.name) for p in pending_batch)
        pending_batch.clear()

    final_completed = list_completed_remote()
    missing = sorted(set(range(1, 201)) - final_completed)

    if missing:
        raise RuntimeError(
            f"Final verification failed. Missing transcripts: {missing}"
        )

    print("FINAL SUCCESS: Episode 0001-0200 TXT transcripts are complete.")
    print(
        f"Hugging Face folder: {HF_OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()
