#!/usr/bin/env python3
import base64, os, re, subprocess, time
from pathlib import Path
from openai import OpenAI
from huggingface_hub import HfApi, hf_hub_download

HF_REPO_ID = "Kumarverma11/PocketFM_Audio"
HF_REPO_TYPE = "dataset"
HF_SOURCE_FOLDER = "Episode_0001_to_0200_Fast"
HF_OUTPUT_FOLDER = "Transcripts_Episode_0001_to_0200"
MODEL = "qwen3.5-omni-plus"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

EPISODES_DIR = Path("episodes")
AUDIO_DIR = Path("/tmp/pocketfm_audio")
REQUEST_DELAY = 2
MAX_RETRIES = 5
RETRY_BASE = 8

HF_TOKEN = os.environ["HF_TOKEN"]
QWEN_API_KEY = os.environ["QWEN_API_KEY"]

EPISODES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

hf = HfApi(token=HF_TOKEN)
client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL, timeout=600)

PROMPT = """इस हिंदी ऑडियो ड्रामा को अत्यंत सटीक रूप से शब्दशः ट्रांसक्राइब करो।
आउटपुट केवल साफ हिंदी टेक्स्ट होना चाहिए, JSON या Markdown नहीं।
कहानी का क्रम, narration, dialogue, नाम, घटनाएँ और episode ending जस की तस रखो।
अपनी ओर से summary, explanation, continuity notes, scene notes या नई कहानी मत जोड़ो।
जहाँ English/Hinglish शब्द वास्तव में बोले गए हों, उन्हें स्वाभाविक रूप में लिखो।
ऑडियो में जो नहीं बोला गया है, वह मत लिखो।
पूरा transcript दो।"""

def episode_number(path):
    m = re.search(r"Episode_(\d{4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None

def git_save_episode(txt_path, ep):
    subprocess.run(["git", "add", str(txt_path)], check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode != 0
    if changed:
        subprocess.run(["git", "commit", "-m", f"Save transcript Episode {ep:04d}"], check=True)
        for attempt in range(5):
            try:
                subprocess.run(["git", "pull", "--rebase"], check=True)
                subprocess.run(["git", "push"], check=True)
                return
            except subprocess.CalledProcessError:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))

def transcribe(audio_path, ep):
    raw = audio_path.read_bytes()
    audio_b64 = base64.b64encode(raw).decode("utf-8")
    fmt = audio_path.suffix.lower().lstrip(".")
    if fmt not in {"wav", "mp3", "ogg", "m4a", "flac"}:
        raise ValueError(f"Unsupported audio format: {fmt}")

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:;base64,{audio_b64}",
                        "format": fmt,
                    },
                },
                {"type": "text", "text": f"Episode {ep}\n\n{PROMPT}"},
            ],
        }],
        modalities=["text"],
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"enable_thinking": True},
    )

    parts = []
    for chunk in stream:
        if chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                parts.append(content)

    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("Qwen returned empty transcript")
    return text

def main():
    files = []
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
            files.append((ep, path))

    files.sort()
    found = [ep for ep, _ in files]
    expected = list(range(1, 201))
    if found != expected:
        missing = sorted(set(expected) - set(found))
        duplicates = sorted({x for x in found if found.count(x) > 1})
        raise RuntimeError(f"Source sequence invalid. Missing={missing}, duplicates={duplicates}")

    print("PASS: Exact Episode 1-200 source sequence found.")

    for index, (ep, repo_path) in enumerate(files, 1):
        txt_path = EPISODES_DIR / f"Episode_{ep:04d}.txt"

        if txt_path.exists() and txt_path.stat().st_size > 50:
            print(f"[{index}/200] Episode {ep:04d} already in GitHub. SKIP")
            continue

        print(f"[{index}/200] Downloading Episode {ep:04d}")
        audio_path = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=repo_path,
            token=HF_TOKEN,
            local_dir=str(AUDIO_DIR),
        ))

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[{index}/200] Qwen transcription attempt {attempt}")
                transcript = transcribe(audio_path, ep)
                txt_path.write_text(transcript + "\n", encoding="utf-8")

                print(f"[{index}/200] Saving Episode {ep:04d} permanently to GitHub")
                git_save_episode(txt_path, ep)
                print(f"[{index}/200] SAVED")
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                wait = RETRY_BASE * (2 ** (attempt - 1))
                print(f"Attempt failed: {exc}")
                if attempt < MAX_RETRIES:
                    print(f"Retry after {wait}s")
                    time.sleep(wait)

        if last_error is not None:
            raise RuntimeError(f"Episode {ep:04d} failed after {MAX_RETRIES} attempts: {last_error}")

        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

        time.sleep(REQUEST_DELAY)

    final_files = sorted(EPISODES_DIR.glob("Episode_*.txt"))
    final_numbers = [episode_number(p.name) for p in final_files]
    if final_numbers != list(range(1, 201)):
        raise RuntimeError("Final transcript sequence is not exactly Episode 1-200.")

    print("PASS: 200 TXT transcripts complete. Uploading full folder to Hugging Face in ONE commit.")
    hf.upload_folder(
        folder_path=str(EPISODES_DIR),
        path_in_repo=HF_OUTPUT_FOLDER,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
        commit_message="Upload complete Episode 0001-0200 TXT transcripts",
    )
    print("FINAL SUCCESS")

if __name__ == "__main__":
    main()
