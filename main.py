#!/usr/bin/env python3
import os,re,json,time
from pathlib import Path
from google import genai
from google.genai import types
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd

HF_TOKEN=os.environ["HF_TOKEN"]; GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]
SRC_REPO=os.getenv("HF_SOURCE_REPO_ID","Kumarverma11/PocketFM_Audio")
SRC_FOLDER=os.getenv("HF_SOURCE_FOLDER","Transcripts_Episode_0001_to_0200")
OUT_REPO=os.getenv("HF_OUTPUT_REPO_ID",SRC_REPO)
OUT_FOLDER=os.getenv("HF_OUTPUT_FOLDER","Veda_Training_Ready_0001_to_0200")
PRIMARY=os.getenv("GEMMA_PRIMARY_MODEL","gemma-4-31b-it")
FALLBACK=os.getenv("GEMMA_FALLBACK_MODEL","gemma-4-26b-a4b-it")
ANALYZER=os.getenv("ANALYSIS_MODEL","gemini-3.1-flash-lite-preview")
BATCH=int(os.getenv("BATCH_SIZE","20")); GAP=float(os.getenv("REQUEST_GAP_SECONDS","5"))
RANGES=((111,120),(121,130),(133,135),(138,139),(140,143))
W=Path("/tmp/veda6"); C=W/"TRACK_A_CLEAN_EPISODES"; I=W/"TRACK_B_STORY_INTELLIGENCE"; D=W/"TRAINING_DATASETS"; S=W/"STATE"
for x in (C,I,D,S): x.mkdir(parents=True,exist_ok=True)
hf=HfApi(token=HF_TOKEN); ai=genai.Client(api_key=GEMINI_API_KEY)

def epno(s):
 m=re.search(r"Episode[_\-\s]*(\d{1,4})",Path(s).name,re.I); return int(m.group(1)) if m else None
def text(resp):
 out=[]
 for c in (getattr(resp,"candidates",None) or []):
  for part in (getattr(getattr(c,"content",None),"parts",None) or []):
   if getattr(part,"text",None): out.append(part.text)
 return "\n".join(out).strip()
def diag(tag,model,r):
 cs=getattr(r,"candidates",None) or []
 print(f"DIAG {tag} model={model} candidates={len(cs)} finish={[str(getattr(c,'finish_reason',None)) for c in cs]} parts={[len(getattr(getattr(c,'content',None),'parts',None) or []) for c in cs]} usage={getattr(r,'usage_metadata',None)}")
def sources():
 pre=SRC_FOLDER.rstrip("/")+"/"; rows=[]
 for x in hf.list_repo_tree(SRC_REPO,repo_type="dataset",recursive=True):
  q=getattr(x,"path",""); n=epno(q)
  if q.startswith(pre) and q.lower().endswith(".txt") and n and 1<=n<=200: rows.append((n,q))
 rows.sort(); nums=[n for n,_ in rows]; miss=sorted(set(range(1,201))-set(nums)); dup=sorted({n for n in nums if nums.count(n)>1})
 if len(rows)!=200 or miss or dup: raise RuntimeError(f"SOURCE INVALID matched={len(rows)} missing={miss} duplicates={dup} first={rows[:10]}")
 return rows
def completed():
 out=set(); pre=f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/"
 try:
  for x in hf.list_repo_tree(OUT_REPO,repo_type="dataset",recursive=True):
   q=getattr(x,"path",""); n=epno(q)
   if q.startswith(pre) and q.endswith(".txt") and n: out.add(n)
 except Exception as e: print("Resume scan:",e)
 return out
def merged(n): return any(a<=n<=b for a,b in RANGES)

def clean_prompt(n,cur,prev,nxt):
 boundary=f"""BOUNDARY REPAIR MODE. This episode came from a previously merged audio range. Compare the previous episode END, current episode, and next episode START. Fix only misplaced or duplicated boundary lines, or a sentence cut at the boundary. Keep chronology. Do not merge whole episodes.
PREVIOUS EPISODE END:
{prev[-6000:]}
NEXT EPISODE START:
{nxt[:6000]}""" if merged(n) else "NORMAL CLEAN MODE. Do not rewrite episode boundaries."
 return f"""You are editing a FICTIONAL HINDI DRAMA AUDIO SERIES transcript.
The existing fictional drama may contain arguments, threats, crime references, fights, injuries, villains, kidnapping references, emotional conflict, or suspense. Treat them as existing fictional plot events. Do not refuse, censor, soften, moralize, or replace an existing scene merely because conflict appears. Do not add graphic detail and do not invent violence.
Your job is ONLY transcript cleanup. Fix Hindi grammar, obvious ASR errors, punctuation, sentence breaks, and clearly wrong names when context proves the correction. Preserve Hindustani/Hinglish, dialogue order, event order, suspense, ending, and cliffhanger.
DO NOT summarize, shorten, add scenes, add dialogue, remove events, explain reasoning, or add headings.
{boundary}
EPISODE {n}
CURRENT RAW TRANSCRIPT:
{cur}
FINAL OUTPUT: Return ONLY the complete corrected current episode transcript. No JSON, no markdown, no explanation."""

def gemma(n,cur,prev,nxt):
 prompt=clean_prompt(n,cur,prev,nxt); errs=[]
 for model in (PRIMARY,FALLBACK):
  for attempt in range(1,4):
   try:
    print(f"CLEAN {n:04d} model={model} attempt={attempt}/3")
    r=ai.models.generate_content(model=model,contents=prompt,config=types.GenerateContentConfig(temperature=.1,max_output_tokens=65536))
    diag(f"EP{n:04d}",model,r); t=text(r)
    if len(t)<200: raise RuntimeError(f"empty/short text chars={len(t)}")
    return t,model
   except Exception as e:
    errs.append(f"{model}#{attempt}:{e}"); print("CLEAN FAIL:",e)
    if attempt<3: time.sleep(8*attempt)
  print("SWITCHING GEMMA MODEL")
 raise RuntimeError(f"EP{n:04d} all Gemma calls failed | "+" | ".join(errs))

SCHEMA={"type":"object","properties":{
"episode":{"type":"integer"},"episode_summary":{"type":"string"},
"character_state":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"state":{"type":"string"},"goal":{"type":"string"},"knowledge":{"type":"string"},"relationship_changes":{"type":"string"}},"required":["name","state","goal","knowledge","relationship_changes"]}},
"plot_threads":{"type":"array","items":{"type":"string"}},"conflict":{"type":"string"},"turning_point":{"type":"string"},
"setup_payoff":{"type":"array","items":{"type":"string"}},"cliffhanger":{"type":"string"},
"continuity_constraints":{"type":"array","items":{"type":"string"}},"next_episode_logic":{"type":"array","items":{"type":"string"}},
"important_facts":{"type":"array","items":{"type":"string"}},"story_learning":{"type":"string"}},
"required":["episode","episode_summary","character_state","plot_threads","conflict","turning_point","setup_payoff","cliffhanger","continuity_constraints","next_episode_logic","important_facts","story_learning"]}

def analyze(n,clean,memory):
 prompt=f"""Analyze this FICTIONAL HINDI DRAMA episode for story-continuity training. Existing fictional conflict, threats, fights, injuries, villains, crime or kidnapping references are plot events; analyze them without refusing or moral commentary. Do not add graphic detail or invent events.
Extract character state, each important character's goal and knowledge, relationship changes, plot threads, conflict, turning point, setup/payoff, cliffhanger, continuity constraints, next-episode logic supported by canon, important facts, and story-writing lesson.
Never invent future canon. Never give a character knowledge they do not have. Track secrets and hidden identity carefully.
EPISODE:{n}
PRIOR MEMORY:{json.dumps(memory,ensure_ascii=False)}
CLEAN TRANSCRIPT:{clean}"""
 for a in range(1,4):
  try:
   print(f"ANALYZE {n:04d} attempt={a}/3")
   r=ai.models.generate_content(model=ANALYZER,contents=prompt,config=types.GenerateContentConfig(temperature=.1,max_output_tokens=16384,response_mime_type="application/json",response_json_schema=SCHEMA,thinking_config=types.ThinkingConfig(thinking_level="high")))
   diag(f"ANALYZE{n:04d}",ANALYZER,r); t=text(r)
   if not t: raise RuntimeError("no text parts")
   z=json.loads(t); z["episode"]=n; return z
  except Exception as e:
   print("ANALYZE FAIL:",e)
   if a<3: time.sleep(8*a)
   else: raise

def save(p,s):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s.strip()+"\n",encoding="utf-8")
def savej(p,z): p.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding="utf-8")
def addjl(p,z):
 with p.open("a",encoding="utf-8") as f:f.write(json.dumps(z,ensure_ascii=False)+"\n")
def memory(old,z,n):
 return {"last_completed_episode":n,"latest_episode_summary":z["episode_summary"],"character_state":z["character_state"],
 "active_plot_threads":list(dict.fromkeys(old.get("active_plot_threads",[])+z["plot_threads"]))[-100:],
 "important_facts":list(dict.fromkeys(old.get("important_facts",[])+z["important_facts"]))[-150:],
 "continuity_constraints":list(dict.fromkeys(old.get("continuity_constraints",[])+z["continuity_constraints"]))[-120:],
 "latest_cliffhanger":z["cliffhanger"],"next_episode_logic":z["next_episode_logic"]}
def upload(pending,msg):
 ops=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in sorted(pending.items())]
 if ops:hf.create_commit(OUT_REPO,repo_type="dataset",operations=ops,commit_message=msg,token=HF_TOKEN)

def main():
 rows=sources(); print("PASS exact 1-200 source folder:",SRC_FOLDER)
 raw={}
 for i,(n,q) in enumerate(rows,1):
  p=hf_hub_download(SRC_REPO,filename=q,repo_type="dataset",token=HF_TOKEN,local_dir=str(W/"raw")); raw[n]=Path(p).read_text(encoding="utf-8",errors="replace").strip()
  if i%25==0: print("Downloaded",i,"/200")
 done=completed(); print("Remote completed:",len(done),"/200")
 ta=D/"track_a_clean_story_text.jsonl"; tb=D/"track_b_story_intelligence.jsonl"; mp=S/"story_memory.json"
 for p in (ta,tb):
  if p.exists():p.unlink()
 mem={}; pending={}; count=0
 for n in range(1,201):
  if n in done: print(f"[{n:03d}/200] REMOTE EXISTS SKIP"); continue
  clean,used=gemma(n,raw[n],raw.get(n-1,""),raw.get(n+1,""))
  cp=C/f"Episode_{n:04d}.txt"; save(cp,clean); time.sleep(GAP)
  z=analyze(n,clean,mem); ip=I/f"Episode_{n:04d}.json"; savej(ip,z)
  addjl(ta,{"episode":n,"text":clean,"clean_model":used}); addjl(tb,z); mem=memory(mem,z,n); savej(mp,mem)
  pending[f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/{cp.name}"]=cp
  pending[f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/{ip.name}"]=ip
  pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{ta.name}"]=ta; pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{tb.name}"]=tb
  pending[f"{OUT_FOLDER}/STATE/story_memory.json"]=mp
  count+=1; print(f"[{n:03d}/200] SUCCESS clean={used} batch={count}/{BATCH}")
  if count>=BATCH:
   upload(pending,f"Veda batch through Episode {n:04d}"); print("BATCH UPLOAD SUCCESS"); pending={}; count=0
  time.sleep(GAP)
 if pending: upload(pending,"Veda final partial batch")
 print("VEDA V6 COMPLETE")
if __name__=="__main__":main()
