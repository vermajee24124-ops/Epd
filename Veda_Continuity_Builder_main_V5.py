#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HF_TOKEN = os.environ["HF_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


def env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


HF_SOURCE_REPO_ID = env_value("HF_SOURCE_REPO_ID", "Kumarverma11/PocketFM_Audio")
HF_SOURCE_REPO_TYPE = env_value("HF_SOURCE_REPO_TYPE", "dataset")
HF_SOURCE_FOLDER = env_value("HF_SOURCE_FOLDER", "Transcripts_Episode_0001_to_0200")

HF_OUTPUT_REPO_ID = env_value("HF_OUTPUT_REPO_ID", "Kumarverma11/Veda_Continuity_Builder")
HF_OUTPUT_REPO_TYPE = env_value("HF_OUTPUT_REPO_TYPE", "dataset")
HF_OUTPUT_FOLDER = env_value("HF_OUTPUT_FOLDER", "Veda_Clean_Analysis")

BATCH_SIZE = int(env_value("BATCH_SIZE", "20"))
REQUEST_DELAY_SECONDS = float(env_value("REQUEST_DELAY_SECONDS", "2.0"))
RETRY_BASE_SECONDS = float(env_value("RETRY_BASE_SECONDS", "8.0"))
MAX_RETRIES = int(env_value("MAX_RETRIES", "4"))

CLEAN_MODEL = env_value("CLEAN_MODEL", "gemma-4-31b-it")
ANALYSIS_MODEL = env_value("ANALYSIS_MODEL", "gemini-3.1-flash-lite")
CLEAN_THINKING_LEVEL = env_value("CLEAN_THINKING_LEVEL", "high")
ANALYSIS_THINKING_LEVEL = env_value("ANALYSIS_THINKING_LEVEL", "high")

WORKDIR = Path(env_value("WORKDIR", "/tmp/veda_continuity_work"))
RAW_DIR = WORKDIR / "raw"
CLEAN_DIR = WORKDIR / "cleaned_episodes"
ANALYSIS_DIR = WORKDIR / "story_intelligence"
BUNDLE_DIR = WORKDIR / "bundle"
STATE_DIR = WORKDIR / "state"
for p in [RAW_DIR, CLEAN_DIR, ANALYSIS_DIR, BUNDLE_DIR, STATE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

hf = HfApi(token=HF_TOKEN)
client = genai.Client(api_key=GEMINI_API_KEY)

SOURCE_EXTS = {".txt"}
MERGED_RANGES = {(111, 120), (121, 130), (133, 135), (138, 139), (140, 143)}


def ep_no(path: str) -> Optional[int]:
    m = re.search(r"Episode_(\d{4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def list_exact_source_texts() -> List[Tuple[int, str]]:
    prefix = HF_SOURCE_FOLDER.rstrip("/") + "/"
    rows: List[Tuple[int, str]] = []
    for item in hf.list_repo_tree(repo_id=HF_SOURCE_REPO_ID, repo_type=HF_SOURCE_REPO_TYPE, recursive=True, expand=False):
        path = getattr(item, "path", "")
        if not path.startswith(prefix):
            continue
        if Path(path).suffix.lower() not in SOURCE_EXTS:
            continue
        ep = ep_no(path)
        if ep and 1 <= ep <= 200:
            rows.append((ep, path))

    rows.sort(key=lambda x: x[0])
    nums = [n for n, _ in rows]
    expected = list(range(1, 201))
    if nums != expected:
        missing = sorted(set(expected) - set(nums))
        dupes = sorted({x for x in nums if nums.count(x) > 1})
        raise RuntimeError(
            f"Source transcript sequence invalid or folder path is wrong. "
            f"Expected only files inside {HF_SOURCE_FOLDER!r}. "
            f"Missing={missing}, duplicates={dupes}. First matched files={rows[:12]}"
        )
    return rows


def completed_remote_episodes() -> set[int]:
    done: set[int] = set()
    try:
        for item in hf.list_repo_tree(repo_id=HF_OUTPUT_REPO_ID, repo_type=HF_OUTPUT_REPO_TYPE, recursive=True, expand=False):
            path = getattr(item, "path", "")
            if not path.startswith(HF_OUTPUT_FOLDER + "/cleaned_episodes/"):
                continue
            if Path(path).suffix.lower() != ".txt":
                continue
            ep = ep_no(path)
            if ep:
                done.add(ep)
    except Exception:
        pass
    return done


def download_raw(repo_path: str) -> Path:
    local = hf_hub_download(
        repo_id=HF_SOURCE_REPO_ID,
        repo_type=HF_SOURCE_REPO_TYPE,
        filename=repo_path,
        token=HF_TOKEN,
        local_dir=str(RAW_DIR),
    )
    return Path(local)


def _clean_text_from_output(text: str) -> Dict:
    cleaned = text
    def collect(section: str) -> List[str]:
        mm = re.search(rf"{section}:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.S | re.I)
        if not mm:
            return []
        block = mm.group(1).strip()
        out = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                line = line[1:].strip()
            if line and line.lower() != "none":
                out.append(line)
        return out

    m = re.search(r"CLEANED_TRANSCRIPT:\s*(.*?)(?:\nCORRECTIONS:|\Z)", text, re.S | re.I)
    if m:
        cleaned = m.group(1).strip()

    return {
        "clean_text": cleaned,
        "corrections": collect("CORRECTIONS"),
        "uncertain_tokens": collect("UNCERTAIN_TOKENS"),
        "boundary_notes": collect("BOUNDARY_NOTES"),
        "raw_model_output": text,
    }


def clean_episode(ep: int, raw_text: str, prev_clean: str, next_raw: str, merged_flag: bool) -> Dict:
    boundary_hint = (
        "This episode comes from a merged source file. Repair any sentence that clearly continues into the next episode. Do not leave a cut-off fragment at the start or end."
        if merged_flag else
        "This episode is a normal single episode."
    )

    prompt = f"""
You are cleaning a Pocket FM style Hindi transcript.

Return plain text with these exact sections:

CLEANED_TRANSCRIPT:
<full cleaned transcript>

CORRECTIONS:
- <short note>
- <short note>

UNCERTAIN_TOKENS:
- <token>
- <token>

BOUNDARY_NOTES:
- <only if a boundary issue was fixed>

Rules:
- Fix grammar, spelling, punctuation, and obvious ASR mistakes.
- Correct names, places, and organizations only when context makes it clear.
- Preserve story order, dialogue order, suspense, and meaning.
- Do NOT summarize.
- Do NOT invent new story.
- Do NOT move events across episodes.
- If a line seems cut off or duplicated because of a merged file boundary, repair it carefully using the previous and next episode as context.
- If no uncertain tokens exist, write "none".
- If no boundary issue exists, write "none".

Episode: {ep}
Boundary hint: {boundary_hint}

PREVIOUS EPISODE CLEAN CONTEXT:
{prev_clean}

CURRENT RAW TRANSCRIPT:
{raw_text}

NEXT EPISODE RAW CONTEXT:
{next_raw}
""".strip()

    resp = client.models.generate_content(
        model=CLEAN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a careful transcript cleaner. Return plain text only. Never rewrite the story.",
            thinking_config=types.ThinkingConfig(thinking_level=CLEAN_THINKING_LEVEL),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemma returned empty output.")
    return _clean_text_from_output(text)


def _analysis_from_text(text: str, ep: int) -> Dict:
    def sec(name: str) -> str:
        m = re.search(rf"{name}:\s*(.*?)(?:\n[A-Z_]+:|\Z)", text, re.S)
        return m.group(1).strip() if m else "none"

    def lines(name: str) -> List[str]:
        block = sec(name)
        if block.lower() == "none":
            return []
        out = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("-"):
                line = line[1:].strip()
            if line:
                out.append(line)
        return out

    char_state_block = sec("CHARACTER_STATE")
    char_state: Dict[str, str] = {}
    if char_state_block.lower() != "none":
        for line in char_state_block.splitlines():
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                char_state[k.strip()] = v.strip()

    return {
        "episode": ep,
        "episode_summary": sec("EPISODE_SUMMARY"),
        "character_state": char_state,
        "plot_threads": lines("PLOT_THREADS"),
        "conflict": sec("CONFLICT"),
        "turning_point": sec("TURNING_POINT"),
        "setup_payoff": lines("SETUP_PAYOFF"),
        "cliffhanger": sec("CLIFFHANGER"),
        "continuity_constraints": lines("CONTINUITY_CONSTRAINTS"),
        "next_episode_logic": lines("NEXT_EPISODE_LOGIC"),
        "important_facts": lines("IMPORTANT_FACTS"),
        "track_a_training_hint": sec("TRACK_A_TRAINING_HINT"),
        "track_b_training_hint": sec("TRACK_B_TRAINING_HINT"),
        "raw_model_output": text,
    }


def gemini_analyze(clean_text: str, ep: int, memory: Dict) -> Dict:
    prompt = f"""
You are analyzing a cleaned episode transcript for continuity training.

Return plain text with these exact sections:

EPISODE_SUMMARY:
<1-3 sentence summary>

CHARACTER_STATE:
<one line per important character in the format name: state>

PLOT_THREADS:
- <active thread>
- <active thread>

CONFLICT:
<main conflict>

TURNING_POINT:
<turning point>

SETUP_PAYOFF:
- <setup/payoff note>

CLIFFHANGER:
<ending hook>

CONTINUITY_CONSTRAINTS:
- <what must stay consistent next>

NEXT_EPISODE_LOGIC:
- <logical next steps>

IMPORTANT_FACTS:
- <fact>
- <fact>

TRACK_A_TRAINING_HINT:
<why this episode matters for text training>

TRACK_B_TRAINING_HINT:
<why this episode matters for reasoning training>

Rules:
- Be concise but useful.
- If a list is empty, write "none".
- Do not add markdown fences.
- Do not add extra commentary.

Current story memory:
{json.dumps(memory, ensure_ascii=False)}

Clean episode transcript:
{clean_text}
""".strip()

    resp = client.models.generate_content(
        model=ANALYSIS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You extract structured continuity data from a clean transcript. Return plain text only.",
            thinking_config=types.ThinkingConfig(thinking_level=ANALYSIS_THINKING_LEVEL),
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned empty output.")
    return _analysis_from_text(text, ep)


def build_memory(ep: int, clean_obj: Dict, analysis_obj: Dict, previous: Dict) -> Dict:
    active_threads = previous.get("active_threads", []) + analysis_obj.get("plot_threads", [])
    important_facts = previous.get("important_facts", []) + analysis_obj.get("important_facts", [])
    continuity_constraints = previous.get("continuity_constraints", []) + analysis_obj.get("continuity_constraints", [])
    return {
        "episode": ep,
        "last_episode_summary": analysis_obj.get("episode_summary", ""),
        "active_threads": list(dict.fromkeys(active_threads))[:120],
        "continuity_constraints": list(dict.fromkeys(continuity_constraints))[:120],
        "important_facts": list(dict.fromkeys(important_facts))[:160],
        "character_state": analysis_obj.get("character_state", {}),
        "clean_preview": clean_obj.get("clean_text", "")[:1200],
        "track_a_hint": analysis_obj.get("track_a_training_hint", ""),
        "track_b_hint": analysis_obj.get("track_b_training_hint", ""),
    }


def upload_batch(pairs: List[Tuple[Path, str]], message: str) -> None:
    seen = set()
    ops = []
    for local_path, remote_path in pairs:
        if remote_path in seen:
            continue
        seen.add(remote_path)
        ops.append(CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=str(local_path)))
    if not ops:
        return
    hf.create_commit(repo_id=HF_OUTPUT_REPO_ID, repo_type=HF_OUTPUT_REPO_TYPE, operations=ops, commit_message=message, token=HF_TOKEN)


def main() -> None:
    rows = list_exact_source_texts()
    already_done = completed_remote_episodes()

    track_a_jsonl = BUNDLE_DIR / "track_a.jsonl"
    track_b_jsonl = BUNDLE_DIR / "track_b.jsonl"
    story_memory_path = BUNDLE_DIR / "story_memory.json"
    manifest_path = BUNDLE_DIR / "manifest.json"

    for p in [track_a_jsonl, track_b_jsonl]:
        if p.exists():
            p.unlink()

    print("PASS: exact source transcript sequence 1-200 found inside the target folder.")
    print(f"Using source folder: {HF_SOURCE_FOLDER}")
    print(f"Already completed on Hugging Face: {len(already_done)}/200")

    raw_cache: Dict[int, str] = {}
    for ep, repo_path in rows:
        raw_cache[ep] = read_text(download_raw(repo_path))

    memory: Dict = {}
    pending_uploads: Dict[str, Path] = {}
    processed_since_upload = 0

    for index, (ep, repo_path) in enumerate(rows, start=1):
        if ep in already_done:
            print(f"[{index:03d}/200] Episode {ep:04d} already uploaded. SKIP")
            continue

        raw_text = raw_cache[ep]
        prev_clean = read_text(CLEAN_DIR / f"Episode_{ep-1:04d}.txt") if ep > 1 and (CLEAN_DIR / f"Episode_{ep-1:04d}.txt").exists() else ""
        next_raw = raw_cache.get(ep + 1, "")
        merged_flag = any(a <= ep <= b for a, b in MERGED_RANGES)

        cleaned_path = CLEAN_DIR / f"Episode_{ep:04d}.txt"
        analysis_path = ANALYSIS_DIR / f"Episode_{ep:04d}.json"

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[{index:03d}/200] CLEAN {ep:04d} | attempt {attempt}/{MAX_RETRIES}")
                clean_obj = clean_episode(ep, raw_text, prev_clean, next_raw, merged_flag)
                clean_text = clean_obj["clean_text"].strip()
                write_text(cleaned_path, clean_text)

                print(f"[{index:03d}/200] ANALYZE {ep:04d} | attempt {attempt}/{MAX_RETRIES}")
                analysis_obj = gemini_analyze(clean_text, ep, memory)
                write_json(analysis_path, analysis_obj)

                append_jsonl(track_a_jsonl, {
                    "episode": ep,
                    "instruction": "Continue the story with continuity.",
                    "input": clean_text,
                    "output_hint": analysis_obj.get("track_a_training_hint", ""),
                    "text": clean_text,
                })
                append_jsonl(track_b_jsonl, analysis_obj)

                memory = build_memory(ep, clean_obj, analysis_obj, memory)
                write_json(story_memory_path, memory)

                pending_uploads[f"{HF_OUTPUT_FOLDER}/cleaned_episodes/{cleaned_path.name}"] = cleaned_path
                pending_uploads[f"{HF_OUTPUT_FOLDER}/story_intelligence/{analysis_path.name}"] = analysis_path
                pending_uploads[f"{HF_OUTPUT_FOLDER}/datasets/track_a.jsonl"] = track_a_jsonl
                pending_uploads[f"{HF_OUTPUT_FOLDER}/datasets/track_b.jsonl"] = track_b_jsonl
                pending_uploads[f"{HF_OUTPUT_FOLDER}/story_memory.json"] = story_memory_path

                processed_since_upload += 1
                break
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                    print(f"  failed: {exc}")
                    print(f"  retry in {wait}s")
                    time.sleep(wait)

        if last_error is not None and not cleaned_path.exists():
            raise RuntimeError(f"Episode {ep:04d} failed after retries: {last_error}")

        if processed_since_upload >= BATCH_SIZE:
            upload_batch(list(pending_uploads.items()), f"Update cleaned transcripts and story intelligence through Episode {ep:04d}")
            print(f"UPLOAD OK through Episode {ep:04d}")
            pending_uploads.clear()
            processed_since_upload = 0

        time.sleep(REQUEST_DELAY_SECONDS)

    if pending_uploads:
        upload_batch(list(pending_uploads.items()), "Finalize cleaned transcripts and story intelligence")

    final_manifest = {
        "project": "Veda Continuity Builder",
        "source_repo": HF_SOURCE_REPO_ID,
        "source_folder": HF_SOURCE_FOLDER,
        "output_repo": HF_OUTPUT_REPO_ID,
        "output_folder": HF_OUTPUT_FOLDER,
        "clean_model": CLEAN_MODEL,
        "analysis_model": ANALYSIS_MODEL,
        "batch_size": BATCH_SIZE,
        "merged_ranges": [f"{a}-{b}" for a, b in sorted(MERGED_RANGES)],
        "completed_remote": sorted(completed_remote_episodes()),
    }
    write_json(manifest_path, final_manifest)
    upload_batch([(manifest_path, f"{HF_OUTPUT_FOLDER}/manifest.json")], "Update manifest")
    print("FINAL SUCCESS: Veda Continuity Builder complete.")


if __name__ == "__main__":
    main()
