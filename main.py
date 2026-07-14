import requests
import json
from datetime import datetime

def fetch_models():
    # OmniRoute का लोकल API एंडपॉइंट जहाँ से मॉडल्स की लिस्ट मिलेगी
    url = "http://localhost:20128/v1/models"
    
    try:
        print(f"Fetching models from {url}...")
        response = requests.get(url)
        response.raise_for_status() # अगर कोई एरर होगा तो यहीं रुक जाएगा
        data = response.json()
        
        # OpenAI स्टैण्डर्ड के अनुसार मॉडल्स 'data' नाम के ऐरे (array) में होते हैं
        models = data.get('data', [])
        
        # आउटपुट फाइल जनरेट करना
        with open("omniroute_models_info.txt", "w", encoding="utf-8") as f:
            f.write("=== OmniRoute Models & Limits Info ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("----------------------------------------\n\n")
            
            if not models:
                f.write("No 'data' array found. Dumping raw JSON instead:\n")
                f.write(json.dumps(data, indent=4))
            else:
                f.write(f"Total Models Found: {len(models)}\n\n")
                for i, model in enumerate(models, 1):
                    model_id = model.get("id", "Unknown ID")
                    owned_by = model.get("owned_by", "Unknown Provider")
                    
                    f.write(f"{i}. Model ID: {model_id}\n")
                    f.write(f"   Provider: {owned_by}\n")
                    
                    # अगर API ने टोकन लिमिट या कोई और एक्स्ट्रा डिटेल दी है, तो उसे यहाँ प्रिंट करेंगे
                    extra_info = {k: v for k, v in model.items() if k not in ["id", "owned_by"]}
                    if extra_info:
                        f.write(f"   Extra Details: {json.dumps(extra_info, indent=2)}\n")
                    
                    f.write("-" * 50 + "\n")
                    
        print("Successfully saved models info to omniroute_models_info.txt")
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        # अगर सर्वर चालू नहीं हुआ या कोई दिक्कत आई, तो एरर को फाइल में लिख देंगे
        with open("omniroute_models_info.txt", "w", encoding="utf-8") as f:
            f.write(f"Failed to fetch models.\nError Details: {e}\n")

if __name__ == "__main__":
    fetch_models()
    
