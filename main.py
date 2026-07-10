#!/usr/bin/env python3
import os,re,json,time
from pathlib import Path
from google import genai
from google.genai import types
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd

HF_TOKEN=os.environ["HF_TOKEN"]
GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]

SRC_REPO=os.getenv("HF_SOURCE_REPO_ID","Kumarverma11/PocketFM_Audio")
SRC_FOLDER=os.getenv("HF_SOURCE_FOLDER","Transcripts_Episode_0001_to_0200")
OUT_REPO=os.getenv("HF_OUTPUT_REPO_ID",SRC_REPO)
OUT_FOLDER=os.getenv("HF_OUTPUT_FOLDER","Veda_Training_Ready_V7_0001_to_0200")

PRIMARY=os.getenv("PRIMARY_MODEL","gemini-3.1-flash-lite")
FALLBACK=os.getenv("FALLBACK_MODEL","gemini-3.5-flash")
BATCH=int(os.getenv("BATCH_SIZE","20"))
GAP=float(os.getenv("REQUEST_GAP_SECONDS","5.0"))
MAX_TRIES=int(os.getenv("MAX_TRIES","4"))

RANGES=((111,120),(121,130),(133,135),(138,139),(140,143))

W=Path("/tmp/veda7")
RAW=W/"raw"
CLEAN=W/"TRACK_A_CLEAN_EPISODES"
INTEL=W/"TRACK_B_STORY_INTELLIGENCE"
DATA=W/"TRAINING_DATASETS"
STATE=W/"STATE"
for d in (RAW,CLEAN,INTEL,DATA,STATE): d.mkdir(parents=True,exist_ok=True)

hf=HfApi(token=HF_TOKEN)
ai=genai.Client(api_key=GEMINI_API_KEY)

def epno(s):
    m=re.search(r"Episode[_\-\s]*(\d{1,4})",Path(s).name,re.I)
    return int(m.group(1)) if m else None

def is_boundary(n):
    return any(a<=n<=b for a,b in RANGES)

def response_text(r):
    chunks=[]
    for c in (getattr(r,"candidates",None) or []):
        content=getattr(c,"content",None)
        for part in (getattr(content,"parts",None) or []):
            t=getattr(part,"text",None)
            if t: chunks.append(t)
    return "\n".join(chunks).strip()

def diagnostics(n,model,r):
    cs=getattr(r,"candidates",None) or []
    print(
        f"DIAG EP{n:04d} model={model} "
        f"candidates={len(cs)} "
        f"finish={[str(getattr(c,'finish_reason',None)) for c in cs]} "
        f"parts={[len(getattr(getattr(c,'content',None),'parts',None) or []) for c in cs]} "
        f"usage={getattr(r,'usage_metadata',None)}"
    )

def source_rows():
    prefix=SRC_FOLDER.rstrip("/")+"/"
    rows=[]
    for x in hf.list_repo_tree(SRC_REPO,repo_type="dataset",recursive=True):
        q=getattr(x,"path","")
        n=epno(q)
        if q.startswith(prefix) and q.lower().endswith(".txt") and n and 1<=n<=200:
            rows.append((n,q))
    rows.sort()
    nums=[n for n,_ in rows]
    missing=sorted(set(range(1,201))-set(nums))
    dup=sorted({n for n in nums if nums.count(n)>1})
    if len(rows)!=200 or missing or dup:
        raise RuntimeError(f"SOURCE INVALID matched={len(rows)} missing={missing} duplicates={dup}")
    return rows

def remote_completed():
    done=set()
    prefix=f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/"
    try:
        for x in hf.list_repo_tree(OUT_REPO,repo_type="dataset",recursive=True):
            q=getattr(x,"path","")
            n=epno(q)
            if q.startswith(prefix) and q.endswith(".txt") and n:
                done.add(n)
    except Exception as e:
        print("Resume scan note:",e)
    return done

def load_remote_memory():
    name=f"{OUT_FOLDER}/STATE/story_memory.json"
    try:
        p=hf_hub_download(OUT_REPO,filename=name,repo_type="dataset",token=HF_TOKEN,local_dir=str(W/"resume"))
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}

SCHEMA={
"type":"object",
"properties":{
"cleaned_transcript":{"type":"string"},
"episode_summary":{"type":"string"},
"character_state":{"type":"array","items":{"type":"object","properties":{
"name":{"type":"string"},"state":{"type":"string"},"goal":{"type":"string"},
"knowledge":{"type":"string"},"relationship_changes":{"type":"string"}},
"required":["name","state","goal","knowledge","relationship_changes"]}},
"plot_threads":{"type":"array","items":{"type":"string"}},
"conflict":{"type":"string"},
"turning_point":{"type":"string"},
"setup_payoff":{"type":"array","items":{"type":"string"}},
"cliffhanger":{"type":"string"},
"continuity_constraints":{"type":"array","items":{"type":"string"}},
"next_episode_logic":{"type":"array","items":{"type":"string"}},
"important_facts":{"type":"array","items":{"type":"string"}},
"story_learning":{"type":"string"}
},
"required":["cleaned_transcript","episode_summary","character_state","plot_threads","conflict",
"turning_point","setup_payoff","cliffhanger","continuity_constraints","next_episode_logic",
"important_facts","story_learning"]
}

def make_prompt(n,raw,prev,nxt,memory):
    if is_boundary(n):
        boundary=f"""
SPECIAL BOUNDARY REPAIR MODE IS ACTIVE because Episode {n} belongs to a range previously split from a merged audio file.
Compare only the end of the previous raw episode, the current raw episode, and the start of the next raw episode.
Repair misplaced or duplicated boundary lines and sentences cut exactly at the boundary.
Do not merge whole episodes and do not move a scene unless the textual continuity clearly proves the cut was wrong.

PREVIOUS RAW EPISODE END:
{prev[-5000:]}

NEXT RAW EPISODE START:
{nxt[:5000]}
"""
    else:
        boundary="NORMAL MODE: Do not perform cross-episode boundary rewriting."

    return f"""
You are processing Episode {n} of a FICTIONAL HINDI DRAMA AUDIO SERIES for machine-learning training data.

FICTION CONTEXT:
The existing story may contain arguments, threats, villains, crime references, fights, injuries, kidnapping references, emotional conflict, or suspense. These are fictional plot events already present in the transcript. Analyze and preserve them as story content. Do not refuse, censor, soften, moralize, or replace an existing fictional scene merely because conflict appears. Do not add graphic detail. Do not invent new violent events.

TASK A — CLEAN TRANSCRIPT:
Fix Hindi grammar, obvious ASR/transcription errors, punctuation, sentence breaks, and clearly wrong character names only when context proves the correction.
Preserve natural Hindustani/Hinglish.
Preserve every real story event, dialogue order, chronology, suspense, ending, and cliffhanger.
Do not summarize or shorten.
Do not rewrite in a new literary style.
Do not add scenes or dialogue.

{boundary}

TASK B — STORY INTELLIGENCE:
Using the cleaned episode and prior memory, extract character states, goals, current knowledge, relationship changes, active plot threads, conflict, turning point, setup/payoff, cliffhanger, continuity constraints, next-episode logic supported by canon, important facts, and story-writing lesson.
Never invent future canon.
Never give a character knowledge the transcript does not show.
Track secrets and hidden identities carefully.

PRIOR STORY MEMORY:
{json.dumps(memory,ensure_ascii=False)}

CURRENT RAW TRANSCRIPT:
{raw}

Return the required structured JSON only.
""".strip()

def process_episode(n,raw,prev,nxt,memory):
    prompt=make_prompt(n,raw,prev,nxt,memory)
    errors=[]
    for model in (PRIMARY,FALLBACK):
        for attempt in range(1,MAX_TRIES+1):
            try:
                print(f"PROCESS EP{n:04d} model={model} attempt={attempt}/{MAX_TRIES}")
                r=ai.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=65536,
                        response_mime_type="application/json",
                        response_json_schema=SCHEMA,
                        thinking_config=types.ThinkingConfig(thinking_level="high"),
                    ),
                )
                diagnostics(n,model,r)
                t=response_text(r)
                if not t:
                    raise RuntimeError("API returned HTTP response but zero text candidates")
                z=json.loads(t)
                if len(z.get("cleaned_transcript","").strip())<200:
                    raise RuntimeError("cleaned transcript is empty or suspiciously short")
                return z,model
            except Exception as e:
                errors.append(f"{model}#{attempt}: {e}")
                print("REQUEST FAILED:",e)
                if attempt<MAX_TRIES:
                    wait=8*attempt
                    print(f"Retry in {wait}s")
                    time.sleep(wait)
        print("MODEL FALLBACK:",model,"-> next model")
    raise RuntimeError(f"Episode {n:04d} failed on all models | "+" | ".join(errors))

def save_text(p,t):
    p.write_text(t.strip()+"\n",encoding="utf-8")

def save_json(p,z):
    p.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding="utf-8")

def append_jsonl(p,z):
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(z,ensure_ascii=False)+"\n")

def update_memory(old,z,n):
    return {
        "last_completed_episode":n,
        "latest_episode_summary":z["episode_summary"],
        "character_state":z["character_state"],
        "active_plot_threads":list(dict.fromkeys(old.get("active_plot_threads",[])+z["plot_threads"]))[-100:],
        "important_facts":list(dict.fromkeys(old.get("important_facts",[])+z["important_facts"]))[-150:],
        "continuity_constraints":list(dict.fromkeys(old.get("continuity_constraints",[])+z["continuity_constraints"]))[-120:],
        "latest_cliffhanger":z["cliffhanger"],
        "next_episode_logic":z["next_episode_logic"],
    }

def upload(pending,msg):
    ops=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in sorted(pending.items())]
    if ops:
        hf.create_commit(repo_id=OUT_REPO,repo_type="dataset",operations=ops,commit_message=msg,token=HF_TOKEN)

def main():
    rows=source_rows()
    print("PASS exact 1-200 source folder:",SRC_FOLDER)
    raw={}
    for i,(n,q) in enumerate(rows,1):
        fp=hf_hub_download(SRC_REPO,filename=q,repo_type="dataset",token=HF_TOKEN,local_dir=str(RAW))
        raw[n]=Path(fp).read_text(encoding="utf-8",errors="replace").strip()
        if i%25==0: print(f"Downloaded {i}/200")

    done=remote_completed()
    memory=load_remote_memory() if done else {}
    print("Remote completed:",len(done),"/200")
    print("Primary:",PRIMARY)
    print("Fallback:",FALLBACK)
    print("Architecture: ONE AI request per episode -> Track A + Track B")

    track_a=DATA/"track_a_clean_story_text.jsonl"
    track_b=DATA/"track_b_story_intelligence.jsonl"
    memory_file=STATE/"story_memory.json"
    pending={}
    batch_count=0

    for n in range(1,201):
        if n in done:
            print(f"[{n:03d}/200] REMOTE EXISTS -> SKIP")
            continue

        z,used=process_episode(n,raw[n],raw.get(n-1,""),raw.get(n+1,""),memory)
        clean=z.pop("cleaned_transcript").strip()
        z["episode"]=n
        z["model_used"]=used

        cp=CLEAN/f"Episode_{n:04d}.txt"
        ip=INTEL/f"Episode_{n:04d}.json"
        save_text(cp,clean)
        save_json(ip,z)
        append_jsonl(track_a,{"episode":n,"text":clean,"model_used":used})
        append_jsonl(track_b,z)

        memory=update_memory(memory,z,n)
        save_json(memory_file,memory)

        pending[f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/{cp.name}"]=cp
        pending[f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/{ip.name}"]=ip
        pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{track_a.name}"]=track_a
        pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{track_b.name}"]=track_b
        pending[f"{OUT_FOLDER}/STATE/story_memory.json"]=memory_file

        batch_count+=1
        print(f"[{n:03d}/200] SUCCESS model={used} batch={batch_count}/{BATCH}")

        if batch_count>=BATCH:
            upload(pending,f"Veda V7 batch through Episode {n:04d}")
            print("BATCH UPLOAD SUCCESS")
            pending={}
            batch_count=0

        time.sleep(GAP)

    if pending:
        upload(pending,"Veda V7 final partial batch")
        print("FINAL BATCH UPLOAD SUCCESS")

    print("VEDA V7 COMPLETE")

if __name__=="__main__":
    main()
