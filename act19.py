from config import api_key
import requests
from PIL import Image
import io
import os
import json

# Define the models
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
TEXT_MODEL = "openai-community/gpt2" 

def query(api_url, payload=None, data=None):
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        if data:
            response = requests.post(api_url, headers=headers, data=data)
        else:
            response = requests.post(api_url, headers=headers, json=payload)
            
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            return None
        return response.content
    except Exception as e:
        print(f"Error while calling API: {e}")
        return None

def get_basic_caption(image):
    print(f"Generating basic caption using {CAPTION_MODEL}...")
    # UPDATED URL: Using the new Router endpoint
    api_url = f"https://router.huggingface.co/hf-inference/models/{CAPTION_MODEL}"
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    
    response_content = query(api_url, data=img_bytes)
    
    if response_content:
        try:
            result = json.loads(response_content.decode("utf-8"))
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No caption generated")
            elif isinstance(result, dict) and "error" in result:
                return f"[Error] {result['error']}"
        except Exception as e:
            return f"[Error] Failed to parse response: {e}"
    return "[Error] API request failed"

def generate_text(prompt, max_new_tokens=60):
    print(f"Generating text with prompt: {prompt}")
    # UPDATED URL: Using the new Router endpoint
    api_url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
    
    response_content = query(api_url, payload=payload)
    
    if response_content:
        try:
            result = json.loads(response_content.decode("utf-8"))
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "error" in result:
                return f"[Error] {result['error']}"
        except Exception as e:
            print(f"Failed to decode text generation response: {e}")
    return ""

def truncate_text(text, word_limit):
    words = text.strip().split()
    return " ".join(words[:word_limit])

def print_menu():
    print(f"""
Select output type:
1. Caption (5 words)
2. Description (30 words)
3. Summary (50 words)
4. Exit
""")

def main():
    image_path = input(f"Enter the path of the image: ")
    if not os.path.exists(image_path):
        print(f"The file '{image_path}' does not exist.")
        return
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    basic_caption = get_basic_caption(image)
    
    if "[Error]" in basic_caption:
        print(f"Captioning failed: {basic_caption}")
        return

    print(f"Basic caption: {basic_caption}")

    while True:
        print_menu()
        choice = input(f"Enter your choice (1-4): ")
        
        if choice == '1':
            caption = truncate_text(basic_caption, 5)
            print(f"Caption (5 words): {caption}")
            
        elif choice == '2':
            prompt_text = f"Expand the following caption into a detailed description: {basic_caption}"
            generated = generate_text(prompt_text, max_new_tokens=60)
            if generated:
                description = truncate_text(generated, 30)
                print(f"Description (30 words): {description}")
                
        elif choice == '3':
            prompt_text = f"Expand the following caption into a detailed summary: {basic_caption}"
            generated = generate_text(prompt_text, max_new_tokens=100)
            if generated:
                summary = truncate_text(generated, 50)
                print(f"Summary (50 words): {summary}")
                
        elif choice == '4':
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
