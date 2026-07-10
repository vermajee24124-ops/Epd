#!/usr/bin/env python3
import os, sys, time, json, re, argparse
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download, HfApi

# ===== CONFIG =====
HF, GQ, NV, GM = os.getenv("HF_TOKEN",""), os.getenv("GROQ_API_KEY",""), os.getenv("NVIDIA_API_KEY",""), os.getenv("GEMINI_API_KEY","")
REPO, INF, OUTF = "Kumarverma11/PocketFM_Audio", "Transcripts_Episode_0001_to_0200", "Veda_Training_Ready_FINAL_0001_to_0200"
A, B, T, S = "TRACK_A_CLEAN_EPISODES", "TRACK_B_STORY_INTELLIGENCE", "TRAINING_DATASETS", "STATE"
BND = {1,2,3,4,5,19,20,21,39,40,41,59,60,61,79,80,81,99,100,101,119,120,121,139,140,141,159,160,161,179,180,181,196,197,198,199,200}
DS, NEMO, GROQ_M = "deepseek-ai/deepseek-v4-pro", "nvidia/nemotron-4-340b-instruct", "llama-3.3-70b-versatile"
BASE = "https://integrate.api.nvidia.com/v1"
ROOT = Path("./outputs"); ROOT.mkdir(exist_ok=True)
for x in [A,B,T,S]: (ROOT/x).mkdir(exist_ok=True)

# ===== HELPERS =====
last = 0
def wait(rpm):
    global last; d=60/rpm; e=time.time()-last
    if e<d: time.sleep(d-e+0.1)
    last=time.time()

def post(url, h, p, t=120):
    try: r=requests.post(url,headers=h,json=p,timeout=t); r.raise_for_status(); return 1,r.json()
    except: return 0,str(sys.exc_info()[1])

def gemini(m, sp, up, mt=8000):
    if not GM: return 0,"",""
    wait(15); ok,d=post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GM}",{},{"contents":[{"role":"user","parts":[{"text":f"{sp}\n\n{up}"}]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":mt,"topP":0.95}})
    return (ok,d["candidates"][0]["content"]["parts"][0]["text"],m) if ok and d.get("candidates") else (0,str(d),m)

def groq(sp, up, mt=8000):
    if not GQ: return 0,"",""
    wait(30); ok,d=post("https://api.groq.com/openai/v1/chat/completions",{"Authorization":f"Bearer {GQ}","Content-Type":"application/json"},{"model":GROQ_M,"messages":[{"role":"system","content":sp},{"role":"user","content":up}],"temperature":0.1,"max_tokens":mt})
    return (ok,d["choices"][0]["message"]["content"],GROQ_M) if ok else (0,d,GROQ_M)

def nvidia(m, sp, up, mt=12000):
    if not NV: return 0,"",""
    wait(30); ok,d=post(f"{BASE}/chat/completions",{"Authorization":f"Bearer {NV}","Content-Type":"application/json"},{"model":m,"messages":[{"role":"system","content":sp},{"role":"user","content":up}],"temperature":0.2,"max_tokens":mt},180)
    return (ok,d["choices"][0]["message"]["content"],m) if ok else (0,d,m)

def py_clean(t):
    for w,r in {"अभ्य":"अभय","शिप्र":"शिप्रा","हाँ":"हां","परन्तु":"परंतु","ज़ोर":"जोर"}.items(): t=t.replace(w,r)
    for p in [r'\b[उअ]h[mह]\b',r'\bhm[mम]\b',r'\b[आअ]h[hह]\b',r'\b[उह]m+\b']: t=re.sub(p,'',t,flags=re.I)
    return '\n'.join(l.strip() for l in t.split('\n') if len(l.strip())>1)

# ===== PROMPTS =====
SA = "You are a Hindi audio drama transcript cleaner. ONLY fix spelling, grammar, punctuation. Remove fillers. Keep speaker labels. Output ONLY cleaned Hindi transcript, no analysis."
PA = "PREVIOUS:\n{prev}\n\nCURRENT (CLEAN THIS):\n{curr}\n\nNEXT:\n{next}\n\nOutput ONLY cleaned transcript."
SB = "You are a story intelligence analyst for Hindi audio dramas. Analyze deeply and return ONLY valid JSON. No markdown, no explanations."
PB = "Analyze this transcript and return JSON.\n\nTRANSCRIPT:\n{t}\n\nMEMORY:\n{m}\n\nReturn JSON with: episode, story_summary, opening_state, character_states, active_plot_threads, conflicts, turning_points, setups, payoffs, continuity_constraints, reveals_and_knowledge, cliffhanger, next_episode_logic, timeline_delta, locations, objects_or_resources, continuity_memory_update, _model."

# ===== CORE =====
def clean_a(txt, ep, prev="", nxt=""):
    up = PA.format(prev=prev[-400:] or "(none)", curr=txt[:15000], next=nxt[:400] or "(none)") if (prev or nxt) else f"CLEAN THIS:\n\n{txt[:15000]}"
    for fn,md in [(lambda s,u: gemini("gemini-2.5-flash-lite",s,u),"gemini"),(lambda s,u: gemini("gemma-4-31b-it",s,u),"gemma"),(lambda s,u: groq(s,u),"groq")]:
        ok,txt2,m=fn(SA,up)
        if ok: return re.sub(r'(?i)^(analysis|summary|note|commentary|here is|cleaned transcript|output)[:\s].*','',txt2).replace('```','').strip(),md
    return py_clean(txt),"python"

def fallback(ep):
    return {"episode":ep,"story_summary":"FAILED","opening_state":{"situation":"?","active_problem":"?","immediate_goal":"?"},"character_states":[],"active_plot_threads":[],"conflicts":[],"turning_points":[],"setups":[],"payoffs":[],"continuity_constraints":[],"reveals_and_knowledge":[],"cliffhanger":{"type":"?","question_created":"?","ending_event":"?","promised_next_pressure":"?"},"next_episode_logic":{"must_continue":[],"likely_immediate_actions":[],"unresolved_questions":[],"do_not_do":[]},"timeline_delta":"?","locations":[],"objects_or_resources":[],"continuity_memory_update":[],"_model":"fallback"}

def analyze_b(txt, ep, mem=""):
    up = PB.format(t=txt[:12000], m=mem[:2000] or "No memory.")
    for md in [DS, NEMO]:
        ok,txt2,m=nvidia(md,SB,up)
        if ok:
            txt2=re.sub(r'```json\s*|```\s*','',txt2).strip()
            try: d=json.loads(txt2); d["_model"]=m; return d,m
            except: return fallback(ep),"json_err"
    return fallback(ep),"fallback"

def dl(ep, cache):
    if ep in cache: return cache[ep]
    try: p=hf_hub_download(repo_id=REPO,filename=f"{INF}/Episode_{ep:04d}.txt",repo_type="dataset"); cache[ep]=Path(p).read_text('utf-8'); return cache[ep]
    except: return ""

def save(ep, folder, ext, content):
    (ROOT/folder/f"Episode_{ep:04d}.{ext}").write_text(content if isinstance(content,str) else json.dumps(content,ensure_ascii=False,indent=2), encoding='utf-8')

def get_mem(ep):
    sp = ROOT/S/"story_memory.json"
    if not sp.exists(): return ""
    mem=json.loads(sp.read_text())
    parts=[f"EP{i}:{mem.get(f'ep_{i:04d}',{}).get('continuity_memory_update',[])}" for i in range(max(1,ep-3),ep)]
    if "_latest" in mem and mem["_latest"].get("last_episode",0)<ep:
        l=mem["_latest"]; parts+=[f"Cliff:{json.dumps(l.get('cliffhanger',{}),ensure_ascii=False)}",f"Facts:{l.get('continuity_facts',[])}"]
    return '\n'.join(parts)[:3000]

def upd_mem(ep, d):
    sp=ROOT/S/"story_memory.json"; mem=json.loads(sp.read_text()) if sp.exists() else {}
    mem[f"ep_{ep:04d}"]={k:d.get(k,[]) for k in ["continuity_constraints","reveals_and_knowledge","active_plot_threads","character_states","next_episode_logic","continuity_memory_update"]}
    mem["_latest"]={"last_episode":ep,"summary":d.get("story_summary",""),"cliffhanger":d.get("cliffhanger",{}),"continuity_facts":[c["fact"] for c in d.get("continuity_constraints",[]) if isinstance(c,dict) and "fact" in c]}
    sp.write_text(json.dumps(mem,ensure_ascii=False,indent=2),encoding='utf-8')

def upload(s,e):
    if not HF: print("  ⚠️ No HF_TOKEN"); return
    hf=HfApi(token=HF)
    for f in [A,B,T,S]:
        local=ROOT/f
        if local.exists() and any(local.iterdir()):
            hf.upload_folder(folder_path=str(local),path_in_repo=f"{OUTF}/{f}",repo_id=REPO,repo_type="dataset",commit_message=f"{f} {s:04d}-{e:04d}")

def run(start,end,a_only,b_only,skip):
    cache={}
    for ep in range(start,end+1):
        print(f"\n🎬 EP{ep:04d}")
        if not b_only:
            raw=dl(ep,cache)
            if not raw: continue
            pv=dl(ep-1,cache)[-400:] if ep in BND else ""
            nx=dl(ep+1,cache)[:400] if ep in BND else ""
            c,m=clean_a(raw,ep,pv,nx); save(ep,A,"txt",c); cache[ep]=c; print(f"  A:{m}")
        if not a_only:
            clean=cache.get(ep) or ((ROOT/A/f"Episode_{ep:04d}.txt").read_text('utf-8') if (ROOT/A/f"Episode_{ep:04d}.txt").exists() else "")
            if not clean: print("  ⏭️ no clean"); continue
            d,m=analyze_b(clean,ep,get_mem(ep)); save(ep,B,"json",d); upd_mem(ep,d)
            save(ep,T,"jsonl",json.dumps({"episode":ep,"input":clean[:5000],"output":json.dumps(d,ensure_ascii=False)[:8000],"task":"story_intelligence"},ensure_ascii=False)+'\n')
            print(f"  B:{m}")
    if not skip: upload(start,end)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--start",type=int,default=1); p.add_argument("--end",type=int,default=200); p.add_argument("--batch-size",type=int,default=20)
    p.add_argument("--track-a-only",action="store_true"); p.add_argument("--track-b-only",action="store_true"); p.add_argument("--skip-upload",action="store_true")
    a=p.parse_args(); print("🎬 PocketFM Pipeline"); s,e=a.start,a.end
    while s<=e:
        be=min(s+a.batch_size-1,e); print(f"\n{'='*40}\nBATCH {s:04d}-{be:04d}\n{'='*40}")
        run(s,be,a.track_a_only,a.track_b_only,a.skip_upload); s=be+1; time.sleep(3)
    print("\n✅ DONE")

if __name__=="__main__": main()
        
