#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HF_TOKEN = os.environ["HF_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

SRC_REPO = os.getenv("HF_SOURCE_REPO_ID", "Kumarverma11/PocketFM_Audio")
SRC_FOLDER = os.getenv("HF_SOURCE_FOLDER", "Transcripts_Episode_0001_to_0200")
OUT_REPO = os.getenv("HF_OUTPUT_REPO_ID", SRC_REPO)
OUT_FOLDER = os.getenv("HF_OUTPUT_FOLDER", "Veda_Training_Ready_V10_0001_to_0200")

# Groq cleaning model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "4500"))
# Your org’s actual free tier may be lower/higher. This is a safe default.
GROQ_TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "10000"))
GROQ_MIN_GAP_SECONDS = float(os.getenv("GROQ_MIN_GAP_SECONDS", "0.2"))

# NVIDIA analysis models
NVIDIA_PRIMARY = os.getenv("NVIDIA_PRIMARY_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NVIDIA_FALLBACK = os.getenv("NVIDIA_FALLBACK_MODEL", "deepseek-ai/deepseek-v4-pro")
NVIDIA_MAX_TOKENS = int(os.getenv("NVIDIA_MAX_TOKENS", "8192"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
GROQ_RETRIES = int(os.getenv("GROQ_RETRIES", "3"))
NVIDIA_RETRIES = int(os.getenv("NVIDIA_RETRIES", "3"))

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

BOUNDARY_RANGES = ((111, 120), (121, 130), (133, 135), (138, 139), (140, 143))

WORK = Path("/tmp/veda10")
RAW = WORK / "raw"
CLEAN = WORK / "TRACK_A_CLEAN_EPISODES"
INTEL = WORK / "TRACK_B_STORY_INTELLIGENCE"
DATA = WORK / "TRAINING_DATASETS"
STATE = WORK / "STATE"
for d in (RAW, CLEAN, INTEL, DATA, STATE):
    d.mkdir(parents=True, exist_ok=True)

hf = HfApi(token=HF_TOKEN)
http = requests.Session()

# Track Groq token usage in a rolling 60s window to stay below TPM safely.
groq_token_events = deque()  # (timestamp, total_tokens)


def epno(s: str) -> Optional[int]:
    m = re.search(r"Episode[_\-\s]*(\d{1,4})", Path(s).name, re.I)
    return int(m.group(1)) if m else None


def in_boundary_range(n: int) -> bool:
    return any(a <= n <= b for a, b in BOUNDARY_RANGES)


def estimate_tokens(text: str) -> int:
    # Rough estimate good enough for pacing decisions.
    return max(1, int(len(text) / 4.0))


def wait_for_groq_budget(next_estimated_tokens: int) -> None:
    now = time.time()
    while groq_token_events and now - groq_token_events[0][0] > 60:
        groq_token_events.popleft()

    used = sum(tokens for _, tokens in groq_token_events)
    if used + next_estimated_tokens <= GROQ_TPM_BUDGET:
        return

    oldest_ts, _ = groq_token_events[0]
    wait = max(1.0, 60 - (now - oldest_ts)) + random.uniform(0.5, 2.0)
    print(f"GROQ budget wait {wait:.1f}s (used={used}, next_est={next_estimated_tokens}, budget={GROQ_TPM_BUDGET})")
    time.sleep(wait)


def call_api(url: str, api_key: str, payload: Dict[str, Any], label: str, retries: int) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            r = http.post(url, headers=headers, json=payload, timeout=(30, 900))
            req_left = r.headers.get("x-ratelimit-remaining-requests")
            tok_left = r.headers.get("x-ratelimit-remaining-tokens")
            print(f"{label} HTTP={r.status_code} try={attempt}/{retries} req_left={req_left} tok_left={tok_left}")

            if r.status_code == 200:
                return r.json()

            if r.status_code in (400, 401, 403, 404, 413):
                raise RuntimeError(f"NON-RETRYABLE HTTP {r.status_code}: {r.text[:1500]}")

            if r.status_code == 429:
                retry_after = (
                    r.headers.get("retry-after")
                    or r.headers.get("x-ratelimit-reset-tokens")
                    or r.headers.get("x-ratelimit-reset-requests")
                )
                wait = parse_wait(retry_after, default=60)
            else:
                wait = min(120, 8 * (2 ** (attempt - 1)))

            wait += random.uniform(0.5, 2.0)
            print(f"{label} retry in {wait:.1f}s")
            time.sleep(wait)

        except requests.RequestException as e:
            last_err = e
            wait = min(120, 8 * (2 ** (attempt - 1))) + random.uniform(0.5, 2.0)
            print(f"{label} network error: {e}; retry in {wait:.1f}s")
            time.sleep(wait)
        except RuntimeError as e:
            last_err = e
            if "NON-RETRYABLE" in str(e):
                raise
            wait = min(90, 8 * attempt) + random.uniform(0.5, 1.5)
            print(f"{label} retryable error: {e}; retry in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"{label} failed after retries: {last_err}")


def parse_wait(value: Optional[str], default: int = 30) -> float:
    if not value:
        return float(default)
    try:
        return max(1.0, float(value))
    except Exception:
        pass

    total = 0.0
    for num, unit in re.findall(r"([\d.]+)\s*(ms|s|m|h)", value.lower()):
        x = float(num)
        if unit == "ms":
            total += x / 1000.0
        elif unit == "s":
            total += x
        elif unit == "m":
            total += x * 60.0
        elif unit == "h":
            total += x * 3600.0
    return max(1.0, total or float(default))


def extract_text_from_openai_json(obj: Dict[str, Any]) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(part.get("text", ""))
            elif isinstance(part, str):
                pieces.append(part)
        return "\n".join(pieces).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def source_rows() -> List[Tuple[int, str]]:
    prefix = SRC_FOLDER.rstrip("/") + "/"
    rows: List[Tuple[int, str]] = []

    for item in hf.list_repo_tree(SRC_REPO, repo_type="dataset", recursive=True):
        path = getattr(item, "path", "")
        n = epno(path)
        if path.startswith(prefix) and path.lower().endswith(".txt") and n and 1 <= n <= 200:
            rows.append((n, path))

    rows.sort()
    nums = [n for n, _ in rows]
    missing = sorted(set(range(1, 201)) - set(nums))
    dup = sorted({n for n in nums if nums.count(n) > 1})

    if len(rows) != 200 or missing or dup:
        raise RuntimeError(f"SOURCE INVALID matched={len(rows)} missing={missing} duplicates={dup}")

    return rows


def remote_completed() -> set[int]:
    done = set()
    prefix = f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/"
    try:
        for item in hf.list_repo_tree(OUT_REPO, repo_type="dataset", recursive=True):
            path = getattr(item, "path", "")
            n = epno(path)
            if path.startswith(prefix) and path.endswith(".json") and n:
                done.add(n)
    except Exception as e:
        print("Resume scan note:", e)
    return done


def download_remote_state() -> Dict[str, Any]:
    remote = f"{OUT_FOLDER}/STATE/story_memory.json"
    try:
        local = hf_hub_download(
            OUT_REPO,
            filename=remote,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(WORK / "resume"),
        )
        return json.loads(Path(local).read_text(encoding="utf-8"))
    except Exception:
        return {}


def groq_prompt(ep: int, current: str, prev: str, nxt: str) -> str:
    if in_boundary_range(ep):
        boundary = f"""
BOUNDARY REPAIR:
Episode {ep} was split from a merged source. Use ONLY the previous ending and next beginning to remove overlap or repair a cut sentence.

PREVIOUS END:
{prev[-3000:]}

NEXT START:
{nxt[:3000]}
"""
    else:
        boundary = "NO BOUNDARY REPAIR NEEDED."

    return f"""
You are editing a fictional Hindi drama transcript.

Keep only the original story. Fix grammar, ASR errors, punctuation, and obvious wrong names.
Do not summarize.
Do not add scenes.
Do not change chronology.
Return only the corrected transcript text.

{boundary}

EPISODE {ep}
CURRENT RAW:
{current}
""".strip()


def groq_clean(ep: int, current: str, prev: str, nxt: str) -> str:
    prompt = groq_prompt(ep, current, prev, nxt)

    # Very short safety estimate so we stay well below TPM.
    next_est = estimate_tokens(prompt) + GROQ_MAX_OUTPUT_TOKENS
    wait_for_groq_budget(next_est)

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": GROQ_MAX_OUTPUT_TOKENS,
        "stream": False,
    }

    errors = []
    for attempt in range(1, GROQ_RETRIES + 1):
        try:
            print(f"GROQ CLEAN EP{ep:04d} attempt={attempt}/{GROQ_RETRIES}")
            obj = call_api(GROQ_URL, GROQ_API_KEY, payload, f"GROQ EP{ep:04d}", GROQ_RETRIES)
            content = extract_text_from_openai_json(obj)

            usage = obj.get("usage") or {}
            total_tokens = int(usage.get("total_tokens") or next_est)
            groq_token_events.append((time.time(), total_tokens))
            while groq_token_events and time.time() - groq_token_events[0][0] > 60:
                groq_token_events.popleft()

            if not content or len(content) < 200:
                raise RuntimeError(f"empty/short Groq content chars={len(content)}")

            return content.strip()

        except Exception as e:
            errors.append(str(e))
            print("GROQ FAIL:", e)
            if attempt < GROQ_RETRIES:
                time.sleep(8 * attempt)

    raise RuntimeError(f"Episode {ep:04d} Groq cleaning failed: {' | '.join(errors)}")


SCHEMA = {
    "type": "object",
    "properties": {
        "episode": {"type": "integer"},
        "episode_summary": {"type": "string"},
        "character_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "state": {"type": "string"},
                    "goal": {"type": "string"},
                    "knowledge": {"type": "string"},
                    "secrets_or_unknowns": {"type": "string"},
                    "relationship_changes": {"type": "string"},
                },
                "required": [
                    "name",
                    "state",
                    "goal",
                    "knowledge",
                    "secrets_or_unknowns",
                    "relationship_changes",
                ],
            },
        },
        "plot_threads": {"type": "array", "items": {"type": "string"}},
        "conflict": {"type": "string"},
        "turning_point": {"type": "string"},
        "setup_payoff": {"type": "array", "items": {"type": "string"}},
        "cliffhanger": {"type": "string"},
        "continuity_constraints": {"type": "array", "items": {"type": "string"}},
        "next_episode_logic": {"type": "array", "items": {"type": "string"}},
        "important_facts": {"type": "array", "items": {"type": "string"}},
        "story_learning": {"type": "string"},
    },
    "required": [
        "episode",
        "episode_summary",
        "character_state",
        "plot_threads",
        "conflict",
        "turning_point",
        "setup_payoff",
        "cliffhanger",
        "continuity_constraints",
        "next_episode_logic",
        "important_facts",
        "story_learning",
    ],
}


def nvidia_prompt(ep: int, clean: str, memory: Dict[str, Any]) -> str:
    return f"""
You are building story-intelligence training data for a fictional Hindi drama series.

Analyze the cleaned episode and the prior memory.
Return only valid JSON matching the schema.

Rules:
- Do not expose chain-of-thought.
- Do not invent future canon.
- Track secrets and hidden identity carefully.
- Keep it concise but useful.
- If a field is empty, use an empty string or an empty list.

Prior memory:
{json.dumps(memory, ensure_ascii=False)}

Clean episode {ep}:
{clean}
""".strip()


def nvidia_analyze(ep: int, clean: str, memory: Dict[str, Any]) -> Dict[str, Any]:
    prompt = nvidia_prompt(ep, clean, memory)
    candidates = [NVIDIA_PRIMARY, NVIDIA_FALLBACK]
    errors = []

    for model in candidates:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": NVIDIA_MAX_TOKENS,
            "stream": False,
        }

        for attempt in range(1, NVIDIA_RETRIES + 1):
            try:
                print(f"NVIDIA ANALYZE EP{ep:04d} model={model} attempt={attempt}/{NVIDIA_RETRIES}")
                obj = call_api(NVIDIA_URL, NVIDIA_API_KEY, payload, f"NVIDIA {model} EP{ep:04d}", NVIDIA_RETRIES)
                txt = extract_text_from_openai_json(obj)
                if not txt:
                    raise RuntimeError("empty NVIDIA content")
                data = extract_json(txt)
                data["episode"] = ep
                data["analysis_model"] = model

                required = [
                    "episode_summary", "character_state", "plot_threads", "conflict",
                    "turning_point", "setup_payoff", "cliffhanger",
                    "continuity_constraints", "next_episode_logic",
                    "important_facts", "story_learning",
                ]
                missing = [k for k in required if k not in data]
                if missing:
                    raise RuntimeError(f"JSON missing keys={missing}")

                return data

            except Exception as e:
                errors.append(f"{model}#{attempt}: {e}")
                print("NVIDIA FAIL:", e)
                if attempt < NVIDIA_RETRIES:
                    time.sleep(8 * attempt)

        print(f"Switching NVIDIA model from {model}")

    raise RuntimeError(f"Episode {ep:04d} NVIDIA analysis failed: {' | '.join(errors)}")


def save_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def update_memory(old: Dict[str, Any], z: Dict[str, Any], ep: int) -> Dict[str, Any]:
    tail = lambda items, limit: list(dict.fromkeys(items))[-limit:]
    return {
        "last_completed_episode": ep,
        "latest_episode_summary": z["episode_summary"],
        "character_state": z["character_state"],
        "active_plot_threads": tail(old.get("active_plot_threads", []) + z["plot_threads"], 100),
        "important_facts": tail(old.get("important_facts", []) + z["important_facts"], 180),
        "continuity_constraints": tail(old.get("continuity_constraints", []) + z["continuity_constraints"], 140),
        "latest_cliffhanger": z["cliffhanger"],
        "next_episode_logic": z["next_episode_logic"],
    }


def restore_completed_files(done: set[int]) -> None:
    for ep in sorted(done):
        for folder, local_dir, ext in [
            ("TRACK_A_CLEAN_EPISODES", CLEAN, "txt"),
            ("TRACK_B_STORY_INTELLIGENCE", INTEL, "json"),
        ]:
            remote = f"{OUT_FOLDER}/{folder}/Episode_{ep:04d}.{ext}"
            try:
                local = hf_hub_download(
                    OUT_REPO,
                    filename=remote,
                    repo_type="dataset",
                    token=HF_TOKEN,
                    local_dir=str(WORK / "restore"),
                )
                (local_dir / f"Episode_{ep:04d}.{ext}").write_bytes(Path(local).read_bytes())
            except Exception as e:
                raise RuntimeError(f"Resume restore failed for {remote}: {e}")


def rebuild_jsonl_from_local() -> Tuple[Path, Path]:
    ta = DATA / "track_a_clean_story_text.jsonl"
    tb = DATA / "track_b_story_intelligence.jsonl"
    with ta.open("w", encoding="utf-8") as a, tb.open("w", encoding="utf-8") as b:
        for ep in range(1, 201):
            cp = CLEAN / f"Episode_{ep:04d}.txt"
            ip = INTEL / f"Episode_{ep:04d}.json"
            if cp.exists():
                a.write(json.dumps({"episode": ep, "text": cp.read_text(encoding="utf-8").strip()}, ensure_ascii=False) + "\n")
            if ip.exists():
                b.write(json.dumps(json.loads(ip.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
    return ta, tb


def upload_batch(batch_eps: List[int], memory: Dict[str, Any]) -> None:
    # Rebuild JSONL files from all locally completed episodes to keep them consistent.
    ta, tb = rebuild_jsonl_from_local()
    mf = STATE / "story_memory.json"
    save_json(mf, memory)

    pending: Dict[str, Path] = {}
    for ep in batch_eps:
        cp = CLEAN / f"Episode_{ep:04d}.txt"
        ip = INTEL / f"Episode_{ep:04d}.json"
        pending[f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/{cp.name}"] = cp
        pending[f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/{ip.name}"] = ip

    pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{ta.name}"] = ta
    pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{tb.name}"] = tb
    pending[f"{OUT_FOLDER}/STATE/story_memory.json"] = mf

    ops = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in sorted(pending.items())
    ]
    hf.create_commit(
        repo_id=OUT_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Veda V10 batch Episodes {batch_eps[0]:04d}-{batch_eps[-1]:04d}",
        token=HF_TOKEN,
    )
    print(f"HF BATCH SUCCESS: {batch_eps[0]:04d}-{batch_eps[-1]:04d}")


def main() -> None:
    rows = source_rows()
    print("PASS source 1-200:", SRC_FOLDER)

    raw: Dict[int, str] = {}
    for i, (ep, repo_path) in enumerate(rows, 1):
        local = hf_hub_download(
            SRC_REPO,
            filename=repo_path,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=str(RAW),
        )
        raw[ep] = Path(local).read_text(encoding="utf-8", errors="replace").strip()
        if i % 25 == 0:
            print(f"Downloaded {i}/200")

    done = remote_completed()
    print(f"Remote complete: {len(done)}/200")
    if done:
        restore_completed_files(done)

    memory = download_remote_state()
    print("TRACK A:", GROQ_MODEL)
    print("TRACK B primary:", NVIDIA_PRIMARY)
    print("TRACK B fallback:", NVIDIA_FALLBACK)

    batch: List[int] = []

    for ep in range(1, 201):
        if ep in done:
            print(f"[{ep:03d}/200] SKIP (already remote)")
            continue

        print(f"\n===== EPISODE {ep:04d} =====")
        clean = groq_clean(ep, raw[ep], raw.get(ep - 1, ""), raw.get(ep + 1, ""))
        if len(clean) < 200:
            raise RuntimeError(f"Groq output too short for EP{ep:04d}")

        cp = CLEAN / f"Episode_{ep:04d}.txt"
        save_text(cp, clean)
        time.sleep(GROQ_MIN_GAP_SECONDS)

        intel = nvidia_analyze(ep, clean, memory)
        ip = INTEL / f"Episode_{ep:04d}.json"
        save_json(ip, intel)
        memory = update_memory(memory, intel, ep)

        batch.append(ep)
        print(f"[{ep:03d}/200] SUCCESS batch={len(batch)}/{BATCH_SIZE}")

        if len(batch) >= BATCH_SIZE:
            upload_batch(batch, memory)
            done.update(batch)
            batch = []

        time.sleep(1.5)

    if batch:
        upload_batch(batch, memory)

    print("VEDA V10 COMPLETE")


if __name__ == "__main__":
    main()
