import os
import json
import time
import glob
import re
from huggingface_hub import HfApi
import google.generativeai as genai
from groq import Groq
from openai import OpenAI

# ================= Configuration =================
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

REPO_ID = "Kumarverma11/PocketFM_Audio"
INPUT_FOLDER = "Transcripts_Episode_0001_to_0200"
OUTPUT_BASE = "Veda_Training_Ready_FINAL_0001_to_0200"

TRACK_A_DIR = os.path.join(OUTPUT_BASE, "TRACK_A_CLEAN_EPISODES")
TRACK_B_DIR = os.path.join(OUTPUT_BASE, "TRACK_B_STORY_INTELLIGENCE")
TRAINING_DIR = os.path.join(OUTPUT_BASE, "TRAINING_DATASETS")
STATE_DIR = os.path.join(OUTPUT_BASE, "STATE")

# Create directories
for d in [TRACK_A_DIR, TRACK_B_DIR, TRAINING_DIR, STATE_DIR]:
    os.makedirs(d, exist_ok=True)

# API Clients Initialization
genai.configure(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
hf_api = HfApi(token=HF_TOKEN)

# ================= Functions =================

def python_fallback_cleanup(text):
    """Fallback: Basic Python string operations for grammar/spelling"""
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    text = text.replace(" ,", ",").replace(" .", ".") # Fix basic punctuation
    # Yahan aap custom dictionary replacements add kar sakte hain
    return text.strip()

def run_track_a_cleanup(text, context_snippet=""):
    """Track A: Clean Transcript (No analysis). Tries Gemini -> Groq -> Python"""
    system_prompt = "You are a transcript cleaner. Output ONLY the corrected transcript. Fix grammar, spelling, and punctuation. Do NOT add any analysis, markdown, or conversational text."
    prompt = f"Context (Do not alter this, use for reference only): {context_snippet}\n\nTranscript to clean:\n{text}"
    
    # Attempt 1: Gemini (Flash)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(system_prompt + "\n\n" + prompt)
        if response.text:
            return response.text.strip()
    except Exception as e:
        print(f"Gemini API failed: {e}. Trying Groq...")

    # Attempt 2: Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192", # Aap apne hisaab se model change kar sakte hain
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=4000
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API failed: {e}. Using Python Fallback...")
    
    # Attempt 3: Python Fallback
    return python_fallback_cleanup(text)


def run_track_b_analysis(cleaned_text, episode_num):
    """Track B: NVIDIA API for High-Thinking JSON extraction (30+ req/min)"""
    system_prompt = """You are a highly advanced Story Intelligence AI. Analyze the given Hindi/Hinglish audio series transcript.
    You MUST output valid JSON EXACTLY matching the provided structure. Do not output anything else.
    Track character states, active plot threads, setups, payoffs, knowledge gaps, cliffhangers, and continuity constraints."""
    
    # Passing a strict instruction to return JSON
    try:
        completion = nvidia_client.chat.completions.create(
            model="meta/llama3-70b-instruct", # Best open model on NVIDIA for logic
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this episode (Number {episode_num}) transcript and return the JSON:\n\n{cleaned_text}"}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"NVIDIA API failed for Episode {episode_num}: {e}")
        return {"episode": episode_num, "error": "Analysis failed"}


def upload_to_huggingface(commit_message):
    """Batch upload the entire output folder to Hugging Face"""
    print(f"Uploading batch to Hugging Face: {commit_message}...")
    try:
        hf_api.upload_folder(
            folder_path=OUTPUT_BASE,
            path_in_repo=OUTPUT_BASE,
            repo_id=REPO_ID,
            commit_message=commit_message
        )
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")

# ================= Main Execution =================

def main():
    # Load all transcripts
    files = sorted(glob.glob(f"{INPUT_FOLDER}/*.txt"))
    if not files:
        print(f"No text files found in {INPUT_FOLDER}")
        return

    episodes_processed = 0

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        episode_num_str = re.search(r'\d+', filename)
        episode_num = int(episode_num_str.group()) if episode_num_str else i+1

        print(f"\n--- Processing {filename} ---")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Handle Context Boundaries (Previous/Next Snippets)
        context = ""
        # Assuming boundary means every 10th or specific structural episodes. 
        # For this example, we fetch previous/next if available just to show the logic.
        # User defined: "agar boundary range me hai to previous/next ka chhota snippet"
        # We will keep it minimal to save tokens as requested.
        is_boundary = (episode_num % 10 == 0) # Example boundary logic
        if is_boundary:
            if i > 0:
                with open(files[i-1], 'r', encoding='utf-8') as prev_f:
                    context += f"PREVIOUS EPISODE SNIPPET: {prev_f.read()[:500]}\n"

        # 1. TRACK A (Cleanup)
        cleaned_text = run_track_a_cleanup(raw_text, context)
        track_a_path = os.path.join(TRACK_A_DIR, f"Cleaned_{filename}")
        with open(track_a_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        # 2. TRACK B (Story Intelligence)
        analysis_json = run_track_b_analysis(cleaned_text, episode_num)
        track_b_path = os.path.join(TRACK_B_DIR, f"Episode_{episode_num:04d}.json")
        with open(track_b_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_json, f, ensure_ascii=False, indent=2)

        # 3. TRAINING DATASET (JSONL)
        jsonl_path = os.path.join(TRAINING_DIR, "training_data.jsonl")
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            training_row = {
                "instruction": f"Generate Episode {episode_num} based on current story state.",
                "input_state": analysis_json.get("opening_state", {}),
                "output_transcript": cleaned_text
            }
            f.write(json.dumps(training_row, ensure_ascii=False) + "\n")

        episodes_processed += 1

        # 4. BATCH UPLOAD (Every 20 Episodes)
        if episodes_processed % 20 == 0:
            upload_to_huggingface(f"Batch upload up to episode {episode_num}")
            # Prevent rate limits just in case
            time.sleep(10)

    # Final upload for any remaining episodes (e.g., if total is not a multiple of 20)
    if episodes_processed % 20 != 0:
        upload_to_huggingface(f"Final batch upload completed")

if __name__ == "__main__":
    main()
