#!/usr/bin/env python3
import os, re, time
from pathlib import Path

import requests
from huggingface_hub import CommitOperationAdd, HfApi
from openai import OpenAI

HF_REPO_ID = "Kumarverma11/PocketFM_Audio"
HF_REPO_TYPE = "dataset"
HF_SOURCE_FOLDER = "Episode_0001_to_0200_Fast"
HF_OUTPUT_FOLDER = "Transcripts_Episode_0001_to_0200"
BATCH_SIZE = 20

# HTTP/file-transcription models only. Realtime WebSocket models are intentionally excluded.
MODELS = ["qwen3.5-omni-plus", "qwen3.5-omni-flash"]

HF_TOKEN = os.environ["HF_TOKEN"]
QWEN_API_KEY = os.environ["QWEN_API_KEY"]

# Existing international compatible endpoint, so only two GitHub secrets are needed.
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

WORK = Path("/tmp/pocketfm")
TXT_DIR = WORK / "episodes"
TXT_DIR.mkdir(parents=True, exist_ok=True)

hf = HfApi(token=HF_TOKEN)
client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, timeout=1800)

PROMPT = """इस हिंदी ऑडियो ड्रामा को पूरा और अत्यंत सटीक शब्दशः ट्रांसक्राइब करो।
केवल साफ transcript text दो। JSON, Markdown, summary, explanation या नई कहानी मत जोड़ो।
Narration, dialogue, character names, Hinglish/English words और घटनाओं का क्रम audio के अनुसार रखो।
कोई हिस्सा मत छोड़ो। Audio में जो नहीं बोला गया है, वह मत लिखो।"""


def ep_no(path):
    m = re.search(r"Episode_(\d{4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def source_files():
    rows = []
    for item in hf.list_repo_tree(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        recursive=True,
        expand=False,
    ):
        path = getattr(item, "path", "")
        ep = ep_no(path)
        if (
            path.startswith(HF_SOURCE_FOLDER + "/")
            and ep
            and 1 <= ep <= 200
            and Path(path).suffix.lower() in {".mp3", ".wav", ".aac", ".m4a"}
        ):
            rows.append((ep, path))
    rows.sort()

    nums = [x[0] for x in rows]
    if nums != list(range(1, 201)):
        missing = sorted(set(range(1, 201)) - set(nums))
        raise RuntimeError(f"Source Episode 1-200 incomplete. Missing={missing}")
    return rows


def completed():
    done = set()
    try:
        for item in hf.list_repo_tree(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            recursive=True,
            expand=False,
        ):
            path = getattr(item, "path", "")
            if path.startswith(HF_OUTPUT_FOLDER + "/") and path.endswith(".txt"):
                ep = ep_no(path)
                if ep:
                    done.add(ep)
    except Exception:
        pass
    return done


def temporary_audio_url(repo_path):
    # Ask private Hugging Face with the token, follow its redirect, and obtain
    # the temporary signed storage URL. Audio bytes are not downloaded by this script.
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{repo_path}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        allow_redirects=True,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    signed_url = response.url
    response.close()
    return signed_url


def transcribe(model, audio_url, audio_format, ep):
    stream = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_url,
                        "format": audio_format,
                    },
                },
                {"type": "text", "text": f"Episode {ep}\n\n{PROMPT}"},
            ],
        }],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
    )

    parts = []
    for chunk in stream:
        if chunk.choices:
            text = chunk.choices[0].delta.content
            if text:
                parts.append(text)

    result = "".join(parts).strip()
    if not result:
        raise RuntimeError("Empty transcript")
    return result


def upload_batch(paths):
    eps = [ep_no(p.name) for p in paths]
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{HF_OUTPUT_FOLDER}/{p.name}",
            path_or_fileobj=str(p),
        )
        for p in paths
    ]

    for attempt in range(1, 6):
        try:
            hf.create_commit(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                operations=operations,
                commit_message=f"Add transcripts Episode {min(eps):04d}-{max(eps):04d}",
                token=HF_TOKEN,
            )
            print(f"UPLOAD OK: Episode {min(eps):04d}-{max(eps):04d}")
            return
        except Exception as e:
            if attempt == 5:
                raise
            wait = 15 * (2 ** (attempt - 1))
            print(f"HF upload retry in {wait}s: {e}")
            time.sleep(wait)


def main():
    rows = source_files()
    done = completed()
    batch = []

    print("SOURCE OK: exact Episode 0001-0200 found")
    print(f"ALREADY UPLOADED: {len(done)}/200")

    for index, (ep, repo_path) in enumerate(rows, 1):
        if ep in done:
            print(f"[{index:03d}/200] Episode {ep:04d} SKIP")
            continue

        print(f"[{index:03d}/200] Episode {ep:04d}: getting temporary audio URL")
        audio_url = temporary_audio_url(repo_path)
        fmt = Path(repo_path).suffix.lower().lstrip(".")
        if fmt == "m4a":
            fmt = "aac"

        transcript = None
        errors = []

        for model in MODELS:
            for attempt in range(1, 4):
                try:
                    print(f"  MODEL {model} | attempt {attempt}/3")
                    transcript = transcribe(model, audio_url, fmt, ep)
                    print(f"  SUCCESS with {model}")
                    break
                except Exception as e:
                    errors.append(f"{model}: {e}")
                    print(f"  FAILED: {e}")
                    if attempt < 3:
                        time.sleep(10 * attempt)
            if transcript:
                break

        if not transcript:
            raise RuntimeError(
                f"Episode {ep:04d} failed on all supported HTTP audio models.\n"
                + "\n".join(errors[-6:])
            )

        txt = TXT_DIR / f"Episode_{ep:04d}.txt"
        txt.write_text(transcript.rstrip() + "\n", encoding="utf-8")
        batch.append(txt)
        print(f"  LOCAL TXT SAVED | batch {len(batch)}/{BATCH_SIZE}")

        if len(batch) == BATCH_SIZE:
            upload_batch(batch)
            for p in batch:
                p.unlink(missing_ok=True)
            batch.clear()

        time.sleep(2)

    if batch:
        upload_batch(batch)

    missing = sorted(set(range(1, 201)) - completed())
    if missing:
        raise RuntimeError(f"Final check failed. Missing={missing}")

    print("FINAL SUCCESS: Episode 0001-0200 TXT complete on Hugging Face")


if __name__ == "__main__":
    main()
