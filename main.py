#!/usr/bin/env python3
import os,re,json,time,random,requests
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,CommitOperationAdd

HF_TOKEN=os.environ["HF_TOKEN"]; GROQ_API_KEY=os.environ["GROQ_API_KEY"]; NVIDIA_API_KEY=os.environ["NVIDIA_API_KEY"]
SRC_REPO=os.getenv("HF_SOURCE_REPO_ID","Kumarverma11/PocketFM_Audio")
SRC_FOLDER=os.getenv("HF_SOURCE_FOLDER","Transcripts_Episode_0001_to_0200")
OUT_REPO=os.getenv("HF_OUTPUT_REPO_ID",SRC_REPO)
OUT_FOLDER=os.getenv("HF_OUTPUT_FOLDER","Veda_Training_Ready_V8_0001_to_0200")
GROQ_MODEL="llama-3.3-70b-versatile"
NV_MODELS=("nvidia/nemotron-3-ultra-550b-a55b","deepseek-ai/deepseek-v4-pro")
BATCH=20; RANGES=((111,120),(121,130),(133,135),(138,139),(140,143))
GROQ_URL="https://api.groq.com/openai/v1/chat/completions"; NV_URL="https://integrate.api.nvidia.com/v1/chat/completions"
W=Path("/tmp/veda8"); RAW=W/"raw"; CLEAN=W/"TRACK_A_CLEAN_EPISODES"; INTEL=W/"TRACK_B_STORY_INTELLIGENCE"; DATA=W/"TRAINING_DATASETS"; STATE=W/"STATE"
for d in (RAW,CLEAN,INTEL,DATA,STATE): d.mkdir(parents=True,exist_ok=True)
hf=HfApi(token=HF_TOKEN); http=requests.Session()

def epno(s):
 m=re.search(r"Episode[_\-\s]*(\d{1,4})",Path(s).name,re.I); return int(m.group(1)) if m else None
def boundary(n): return any(a<=n<=b for a,b in RANGES)
def wait_value(v,default=30):
 if not v:return default
 try:return max(1,float(v))
 except:pass
 total=0
 for x,u in re.findall(r"([\d.]+)\s*(ms|s|m|h)",v.lower()):
  x=float(x); total+=x/1000 if u=="ms" else x if u=="s" else x*60 if u=="m" else x*3600
 return max(1,total or default)
def call(url,key,payload,label,tries=6):
 h={"Authorization":f"Bearer {key}","Content-Type":"application/json"}; last=None
 for a in range(1,tries+1):
  try:
   r=http.post(url,headers=h,json=payload,timeout=(30,900))
   print(f"{label} HTTP={r.status_code} try={a}/{tries} req_left={r.headers.get('x-ratelimit-remaining-requests')} tok_left={r.headers.get('x-ratelimit-remaining-tokens')}")
   if r.status_code==200:
    z=r.json(); ch=z.get("choices") or []
    if ch and ((ch[0].get("message") or {}).get("content") or "").strip(): return ch[0]["message"]["content"].strip()
    last=RuntimeError("HTTP 200 but empty choices/content")
   elif r.status_code in (400,401,403,404): raise RuntimeError(f"NON-RETRYABLE HTTP {r.status_code}: {r.text[:1000]}")
   else: last=RuntimeError(f"HTTP {r.status_code}: {r.text[:1000]}")
   if r.status_code==429: delay=wait_value(r.headers.get("retry-after") or r.headers.get("x-ratelimit-reset-tokens") or r.headers.get("x-ratelimit-reset-requests"),60)
   else: delay=min(120,8*(2**(a-1)))
  except requests.RequestException as e: last=e; delay=min(120,8*(2**(a-1)))
  if a<tries:
   delay+=random.uniform(.5,2); print(f"{label} retry in {delay:.1f}s"); time.sleep(delay)
 raise RuntimeError(f"{label} failed: {last}")
def json_obj(t):
 t=t.strip(); t=re.sub(r"^```(?:json)?\s*","",t,flags=re.I); t=re.sub(r"\s*```$","",t)
 try:return json.loads(t)
 except:return json.loads(t[t.find("{"):t.rfind("}")+1])
def rows():
 pre=SRC_FOLDER.rstrip("/")+"/"; out=[]
 for x in hf.list_repo_tree(SRC_REPO,repo_type="dataset",recursive=True):
  q=getattr(x,"path",""); n=epno(q)
  if q.startswith(pre) and q.lower().endswith(".txt") and n and 1<=n<=200:out.append((n,q))
 out.sort(); nums=[n for n,_ in out]; miss=sorted(set(range(1,201))-set(nums)); dup=sorted({n for n in nums if nums.count(n)>1})
 if len(out)!=200 or miss or dup:raise RuntimeError(f"SOURCE INVALID count={len(out)} missing={miss} duplicates={dup}")
 return out
def done_set():
 out=set(); pre=f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/"
 try:
  for x in hf.list_repo_tree(OUT_REPO,repo_type="dataset",recursive=True):
   q=getattr(x,"path",""); n=epno(q)
   if q.startswith(pre) and q.endswith(".json") and n:out.add(n)
 except Exception as e:print("Resume scan:",e)
 return out
def remote_json(path):
 try:
  p=hf_hub_download(OUT_REPO,filename=path,repo_type="dataset",token=HF_TOKEN,local_dir=str(W/"resume")); return json.loads(Path(p).read_text(encoding="utf-8"))
 except:return {}
def clean(n,cur,prev,nxt):
 b=f"""SPECIAL BOUNDARY REPAIR. Episode {n} came from a previously merged audio file. Compare PREVIOUS END, CURRENT and NEXT START. Remove from CURRENT only lines clearly belonging to the adjacent episode. Repair only a sentence cut exactly at the boundary. Do not merge episodes or invent scenes.
PREVIOUS END:\n{prev[-5000:]}\nNEXT START:\n{nxt[:5000]}""" if boundary(n) else "NORMAL MODE. Do not rewrite episode boundaries."
 prompt=f"""You are a precision transcript editor for a FICTIONAL HINDI DRAMA AUDIO SERIES. Existing fictional arguments, threats, villains, crime references, fights, injuries, kidnapping references, emotional conflict and suspense are plot events. Preserve them. Do not moralize, censor, summarize, or add graphic detail.
Fix obvious Hindi grammar, ASR errors, punctuation, sentence breaks, and clearly wrong character names only when context proves the correction. Preserve natural Hindi/Hindustani/Hinglish, every event, dialogue order, chronology, suspense, ending and cliffhanger. Do not shorten, summarize, add scenes, or add dialogue. Return ONLY the full cleaned transcript.
{b}
EPISODE {n}\nCURRENT RAW TRANSCRIPT:\n{cur}"""
 return call(GROQ_URL,GROQ_API_KEY,{"model":GROQ_MODEL,"messages":[{"role":"user","content":prompt}],"temperature":0.05,"max_tokens":32768,"stream":False},f"GROQ EP{n:04d}")
def analyze(n,txt,memory):
 shape='{"episode":1,"episode_summary":"string","character_state":[{"name":"string","state":"string","goal":"string","knowledge":"string","secrets_or_unknowns":"string","relationship_changes":"string"}],"plot_threads":["string"],"conflict":"string","turning_point":"string","setup_payoff":["string"],"cliffhanger":"string","continuity_constraints":["string"],"next_episode_logic":["string"],"important_facts":["string"],"story_learning":"string"}'
 prompt=f"""Analyze Episode {n} of a FICTIONAL HINDI DRAMA for story-continuity training. Reason deeply internally but return ONLY valid JSON and never expose chain-of-thought. Track character state, goal, knowledge, secrets, hidden identity, relationship changes, plot threads, conflict, turning point, setup/payoff, cliffhanger, continuity constraints, important facts and canon-supported next-episode logic. Never invent future canon or give a character knowledge they do not possess.
JSON SHAPE:{shape}
PRIOR MEMORY:{json.dumps(memory,ensure_ascii=False)}
CLEAN EPISODE:{txt}"""
 errs=[]
 for model in NV_MODELS:
  try:
   out=call(NV_URL,NVIDIA_API_KEY,{"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.2,"max_tokens":16384,"stream":False},f"NVIDIA {model} EP{n:04d}")
   z=json_obj(out); required=("episode_summary","character_state","plot_threads","conflict","turning_point","setup_payoff","cliffhanger","continuity_constraints","next_episode_logic","important_facts","story_learning")
   miss=[k for k in required if k not in z]
   if miss:raise RuntimeError(f"JSON missing {miss}")
   z["episode"]=n; z["analysis_model"]=model; return z
  except Exception as e:errs.append(f"{model}: {e}"); print("NVIDIA MODEL FAIL:",e)
 raise RuntimeError("All NVIDIA models failed | "+" | ".join(errs))
def savej(p,z):p.write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding="utf-8")
def memory(old,z,n):
 u=lambda a,l:list(dict.fromkeys(a))[-l:]
 return {"last_completed_episode":n,"latest_episode_summary":z["episode_summary"],"character_state":z["character_state"],"active_plot_threads":u(old.get("active_plot_threads",[])+z["plot_threads"],100),"important_facts":u(old.get("important_facts",[])+z["important_facts"],180),"continuity_constraints":u(old.get("continuity_constraints",[])+z["continuity_constraints"],140),"latest_cliffhanger":z["cliffhanger"],"next_episode_logic":z["next_episode_logic"]}
def restore(done):
 for n in sorted(done):
  for folder,local,ext in (("TRACK_A_CLEAN_EPISODES",CLEAN,"txt"),("TRACK_B_STORY_INTELLIGENCE",INTEL,"json")):
   q=f"{OUT_FOLDER}/{folder}/Episode_{n:04d}.{ext}"; p=hf_hub_download(OUT_REPO,filename=q,repo_type="dataset",token=HF_TOKEN,local_dir=str(W/"restore")); (local/f"Episode_{n:04d}.{ext}").write_bytes(Path(p).read_bytes())
def upload(batch,mem):
 ta=DATA/"track_a_clean_story_text.jsonl"; tb=DATA/"track_b_story_intelligence.jsonl"
 complete=[n for n in range(1,201) if (CLEAN/f"Episode_{n:04d}.txt").exists() and (INTEL/f"Episode_{n:04d}.json").exists()]
 with ta.open("w",encoding="utf-8") as a,tb.open("w",encoding="utf-8") as b:
  for n in complete:
   a.write(json.dumps({"episode":n,"text":(CLEAN/f"Episode_{n:04d}.txt").read_text(encoding="utf-8").strip()},ensure_ascii=False)+"\n")
   b.write(json.dumps(json.loads((INTEL/f"Episode_{n:04d}.json").read_text(encoding="utf-8")),ensure_ascii=False)+"\n")
 mf=STATE/"story_memory.json";savej(mf,mem); pending={}
 for n in batch:
  pending[f"{OUT_FOLDER}/TRACK_A_CLEAN_EPISODES/Episode_{n:04d}.txt"]=CLEAN/f"Episode_{n:04d}.txt";pending[f"{OUT_FOLDER}/TRACK_B_STORY_INTELLIGENCE/Episode_{n:04d}.json"]=INTEL/f"Episode_{n:04d}.json"
 pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{ta.name}"]=ta;pending[f"{OUT_FOLDER}/TRAINING_DATASETS/{tb.name}"]=tb;pending[f"{OUT_FOLDER}/STATE/story_memory.json"]=mf
 ops=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=str(v)) for k,v in sorted(pending.items())]
 hf.create_commit(repo_id=OUT_REPO,repo_type="dataset",operations=ops,commit_message=f"Veda V8 Episodes {batch[0]:04d}-{batch[-1]:04d}",token=HF_TOKEN);print("HF BATCH SUCCESS",batch)
def main():
 src=rows();print("PASS source 1-200:",SRC_FOLDER);raw={}
 for i,(n,q) in enumerate(src,1):
  p=hf_hub_download(SRC_REPO,filename=q,repo_type="dataset",token=HF_TOKEN,local_dir=str(RAW));raw[n]=Path(p).read_text(encoding="utf-8",errors="replace").strip()
  if i%25==0:print(f"Downloaded {i}/200")
 done=done_set();print(f"Remote complete {len(done)}/200")
 if done:restore(done)
 mem=remote_json(f"{OUT_FOLDER}/STATE/story_memory.json");batch=[]
 print("TRACK A Groq:",GROQ_MODEL);print("TRACK B NVIDIA:",NV_MODELS);print("Adaptive 429 Retry-After enabled")
 for n in range(1,201):
  if n in done:print(f"[{n:03d}/200] SKIP");continue
  print(f"===== EPISODE {n:04d} =====")
  txt=clean(n,raw[n],raw.get(n-1,""),raw.get(n+1,""))
  if len(txt)<200:raise RuntimeError(f"Groq output too short EP{n}")
  cp=CLEAN/f"Episode_{n:04d}.txt";cp.write_text(txt.strip()+"\n",encoding="utf-8");time.sleep(6)
  z=analyze(n,txt,mem);ip=INTEL/f"Episode_{n:04d}.json";savej(ip,z);mem=memory(mem,z,n);time.sleep(2)
  batch.append(n);print(f"[{n:03d}/200] COMPLETE batch={len(batch)}/{BATCH}")
  if len(batch)>=BATCH:upload(batch,mem);done.update(batch);batch=[]
 if batch:upload(batch,mem)
 print("VEDA V8 COMPLETE")
if __name__=="__main__":main()
