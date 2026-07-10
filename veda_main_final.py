#!/usr/bin/env python3
from __future__ import annotations
import json, os, re, time, unicodedata
from pathlib import Path
from typing import Any
import requests
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

HF_REPO = "Kumarverma11/PocketFM_Audio"
HF_TYPE = "dataset"
SOURCE_FOLDER = "Transcripts_Episode_0001_to_0200"
OUTPUT_FOLDER = "Veda_Training_Ready_FINAL_0001_to_0200"
TRACK_A = f"{OUTPUT_FOLDER}/TRACK_A_CLEAN_EPISODES"
TRACK_B = f"{OUTPUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE"
DATASETS = f"{OUTPUT_FOLDER}/TRAINING_DATASETS"
STATE = f"{OUTPUT_FOLDER}/STATE"
BATCH_SIZE = 20

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
NVIDIA_MODELS = [
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "deepseek-ai/deepseek-r1",
]

MERGED_RANGES = [(111,120),(121,130),(133,135),(138,139),(140,143)]
WORK = Path("/tmp/veda_final")
RAW = WORK/"raw"; CLEAN = WORK/"clean"; INTEL = WORK/"intel"
for p in (RAW,CLEAN,INTEL): p.mkdir(parents=True, exist_ok=True)

def secret(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v: raise RuntimeError(f"Missing GitHub Secret: {name}")
    return v

HF_TOKEN = secret("HF_TOKEN")
GROQ_API_KEY = secret("GROQ_API_KEY")
NVIDIA_API_KEY = secret("NVIDIA_API_KEY")
api = HfApi(token=HF_TOKEN)

def epnum(path: str):
    m = re.search(r"episode[_\s-]*0*(\d{1,4})", Path(path).name, re.I)
    return int(m.group(1)) if m else None

def clean_python(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff","")
    text = text.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([।!?])\1{1,}", r"\1", text)
    return text.strip()

def remote_paths():
    return [x.path for x in api.list_repo_tree(HF_REPO, repo_type=HF_TYPE, recursive=True)]

def source_map(paths):
    out = {}
    prefix = SOURCE_FOLDER + "/"
    for p in paths:
        if p.startswith(prefix) and p.lower().endswith(".txt"):
            n = epnum(p)
            if n and 1 <= n <= 200:
                if n in out: raise RuntimeError(f"Duplicate source episode {n}: {out[n]} AND {p}")
                out[n] = p
    missing = [n for n in range(1,201) if n not in out]
    if missing: raise RuntimeError(f"Source missing episodes: {missing}")
    return out

def chat(url, key, model, messages, max_tokens, temperature=0.1, retries=5):
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}
    payload={"model":model,"messages":messages,"temperature":temperature,
             "max_tokens":max_tokens,"stream":False}
    last=""
    for attempt in range(retries):
        try:
            r=requests.post(url,headers=headers,json=payload,timeout=300)
            if r.status_code in (429,500,502,503,504):
                wait=min(90, 8*(2**attempt))
                print(f"HTTP {r.status_code}; retry in {wait}s")
                time.sleep(wait); continue
            r.raise_for_status()
            data=r.json()
            content=data["choices"][0]["message"].get("content")
            if isinstance(content,list):
                content="".join(x.get("text","") if isinstance(x,dict) else str(x) for x in content)
            if content and content.strip(): return content.strip()
            last=f"empty response: {str(data)[:800]}"
        except Exception as e:
            last=str(e)
        time.sleep(min(90,8*(2**attempt)))
    raise RuntimeError(f"{model} failed: {last}")

def parse_json(text: str) -> dict:
    text=text.strip()
    text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I)
    try: return json.loads(text)
    except Exception:
        a=text.find("{"); b=text.rfind("}")
        if a>=0 and b>a: return json.loads(text[a:b+1])
        raise

def in_merged(n:int)->bool:
    return any(a<=n<=b for a,b in MERGED_RANGES)

def repair_boundary(prev_text:str, cur_text:str, prev_n:int, cur_n:int):
    tail=prev_text[-5000:]
    head=cur_text[:5000]
    prompt=f"""Hindi audio drama. Episodes {prev_n} and {cur_n} came from a split merged audio.
Check ONLY the boundary. Some ending lines may be in the wrong episode.
Do not rewrite, summarize, censor, or improve the story.
Return JSON only:
{{"left_tail":"corrected tail for episode {prev_n}","right_head":"corrected head for episode {cur_n}","changed":true}}
LEFT_TAIL:
{tail}
RIGHT_HEAD:
{head}"""
    raw=chat(GROQ_URL,GROQ_API_KEY,GROQ_MODEL,
             [{"role":"system","content":"You repair transcript boundaries. Fictional drama text is allowed. Preserve original Hindi wording. JSON only."},
              {"role":"user","content":prompt}], max_tokens=4200)
    obj=parse_json(raw)
    lt=str(obj.get("left_tail","")).strip()
    rh=str(obj.get("right_head","")).strip()
    if not lt or not rh: raise RuntimeError(f"Boundary JSON invalid for {prev_n}-{cur_n}")
    return prev_text[:-len(tail)] + lt, rh + cur_text[len(head):], bool(obj.get("changed",False))

SCHEMA = {
 "episode": 1,
 "story_summary": "",
 "opening_state": {"situation":"","active_problem":"","immediate_goal":""},
 "character_states": [{"name":"","role":"","current_goal":"","emotion":"","knowledge":[],"relationships":[],"change_in_episode":""}],
 "active_plot_threads": [{"thread":"","status":"opened|advanced|paused|resolved","evidence":"","next_pressure":""}],
 "conflicts": [{"type":"internal|interpersonal|external","characters":[],"cause":"","development":"","result":""}],
 "turning_points": [{"event":"","before":"","after":"","why_it_matters":""}],
 "setups": [{"setup":"","possible_payoff":"","status":"new|carried"}],
 "payoffs": [{"payoff":"","setup_reference":"","effect":""}],
 "continuity_constraints": [{"fact":"","must_remain_true_until_changed":"","risk_if_ignored":""}],
 "reveals_and_knowledge": [{"fact":"","known_by":[],"unknown_to":[],"effect":""}],
 "cliffhanger": {"type":"","question_created":"","ending_event":"","promised_next_pressure":""},
 "next_episode_logic": {"must_continue":[],"likely_immediate_actions":[],"unresolved_questions":[],"do_not_do":[]},
 "timeline_delta": "",
 "locations": [],
 "objects_or_resources": [],
 "continuity_memory_update": []
}

def story_intelligence(n:int, text:str, previous_memory:list[dict]):
    memory=json.dumps(previous_memory[-30:],ensure_ascii=False)
    schema=json.dumps(SCHEMA,ensure_ascii=False)
    user=f"""Episode {n}. Extract training intelligence from this Hindi drama transcript.
Use only facts supported by the transcript. No future invention. JSON only.
Recent continuity memory:
{memory}
Required JSON shape:
{schema}
TRANSCRIPT:
{text}"""
    last=""
    for model in NVIDIA_MODELS:
        try:
            raw=chat(NVIDIA_URL,NVIDIA_API_KEY,model,
                     [{"role":"system","content":"You are a continuity analyst for long Hindi audio drama. Think carefully internally. Output valid JSON only. Never add facts not present in the episode."},
                      {"role":"user","content":user}], max_tokens=6500, retries=3)
            obj=parse_json(raw)
            obj["episode"]=n
            obj["_model"]=model
            for key in ("story_summary","character_states","active_plot_threads","continuity_constraints","cliffhanger","next_episode_logic"):
                if key not in obj: raise ValueError(f"missing key {key}")
            return obj
        except Exception as e:
            last=f"{model}: {e}"
            print("NVIDIA fallback:", last)
    raise RuntimeError("All NVIDIA models failed. Last="+last)

def write_json(path:Path,obj:Any):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def upload_batch(start:int,end:int, memory:list[dict]):
    ops=[]
    for n in range(start,end+1):
        a=CLEAN/f"Episode_{n:04d}.txt"
        b=INTEL/f"Episode_{n:04d}.json"
        if a.exists(): ops.append(CommitOperationAdd(path_in_repo=f"{TRACK_A}/{a.name}",path_or_fileobj=str(a)))
        if b.exists(): ops.append(CommitOperationAdd(path_in_repo=f"{TRACK_B}/{b.name}",path_or_fileobj=str(b)))
    # batch JSONL files are training-ready and append-safe by range
    text_jsonl=WORK/f"track_a_{start:04d}_{end:04d}.jsonl"
    intel_jsonl=WORK/f"track_b_{start:04d}_{end:04d}.jsonl"
    with text_jsonl.open("w",encoding="utf-8") as fa, intel_jsonl.open("w",encoding="utf-8") as fb:
        for n in range(start,end+1):
            ap=CLEAN/f"Episode_{n:04d}.txt"; bp=INTEL/f"Episode_{n:04d}.json"
            if ap.exists(): fa.write(json.dumps({"episode":n,"text":ap.read_text(encoding="utf-8")},ensure_ascii=False)+"\n")
            if bp.exists(): fb.write(json.dumps(json.loads(bp.read_text(encoding="utf-8")),ensure_ascii=False)+"\n")
    ops += [
      CommitOperationAdd(path_in_repo=f"{DATASETS}/{text_jsonl.name}",path_or_fileobj=str(text_jsonl)),
      CommitOperationAdd(path_in_repo=f"{DATASETS}/{intel_jsonl.name}",path_or_fileobj=str(intel_jsonl)),
    ]
    state_file=WORK/"story_memory.json"
    write_json(state_file,{"completed_through":end,"memory":memory[-100:]})
    ops.append(CommitOperationAdd(path_in_repo=f"{STATE}/story_memory.json",path_or_fileobj=str(state_file)))
    api.create_commit(repo_id=HF_REPO,repo_type=HF_TYPE,operations=ops,
                      commit_message=f"Veda episodes {start:04d}-{end:04d}",token=HF_TOKEN)
    print(f"UPLOADED ONE COMMIT: {start:04d}-{end:04d}")

def main():
    paths=remote_paths()
    src=source_map(paths)
    done=set()
    for p in paths:
        if p.startswith(TRACK_B+"/") and p.endswith(".json"):
            n=epnum(p)
            if n: done.add(n)
    print(f"PASS source 1-200 | remote complete {len(done)}/200")
    memory=[]
    state_path=f"{STATE}/story_memory.json"
    if state_path in paths:
        local=hf_hub_download(HF_REPO,state_path,repo_type=HF_TYPE,token=HF_TOKEN)
        memory=json.loads(Path(local).read_text(encoding="utf-8")).get("memory",[])
    batch_start=None
    prev_n=None; prev_clean=None
    for n in range(1,201):
        if n in done:
            continue
        local=hf_hub_download(HF_REPO,src[n],repo_type=HF_TYPE,token=HF_TOKEN,local_dir=str(RAW))
        raw=Path(local).read_text(encoding="utf-8",errors="replace")
        current=clean_python(raw)
        # Only boundaries inside the user-confirmed merged ranges use Groq.
        if prev_n is not None and in_merged(prev_n) and in_merged(n):
            prev_clean,current,changed=repair_boundary(prev_clean,current,prev_n,n)
            (CLEAN/f"Episode_{prev_n:04d}.txt").write_text(prev_clean,encoding="utf-8")
            print(f"BOUNDARY {prev_n}-{n} checked changed={changed}")
        (CLEAN/f"Episode_{n:04d}.txt").write_text(current,encoding="utf-8")
        intel=story_intelligence(n,current,memory)
        write_json(INTEL/f"Episode_{n:04d}.json",intel)
        memory.extend(intel.get("continuity_memory_update",[]))
        if not memory:
            memory.append({"episode":n,"summary":intel.get("story_summary","")})
        print(f"DONE {n:04d}/0200")
        if batch_start is None: batch_start=n
        prev_n=n; prev_clean=current
        # Upload every 20 processed episodes, or at episode 200.
        made=len([p for p in INTEL.glob("Episode_*.json")])
        if made % BATCH_SIZE == 0 or n==200:
            batch_eps=sorted(epnum(p.name) for p in INTEL.glob("Episode_*.json"))
            pending=[x for x in batch_eps if x not in done]
            if pending:
                upload_batch(min(pending),max(pending),memory)
                done.update(pending)
        time.sleep(0.5)
    print("VEDA FINAL COMPLETE")

if __name__=="__main__":
    main()
