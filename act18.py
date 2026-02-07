from config import api_key
import requests
from PIL import Image
import io
import os
from colorama import init, Fore, Style
import json
init(autoreset=True)

def query(api_url,payload=None, files=None, method="post"):
    headers={"Authorization": f"Bearer {api_key}"}
    try:
        if method.lower()=="post":
            response=requests.post(api_url, headers=headers, json=payload, files=files)
        else:
            response=requests.get(api_url, headers=headers, params=payload)
        if response.status_code != 200:
             raise Exception (f"Status {response.status_code}: {response.text}")
        return response.content
    except Exception as e:
        print(f"{Fore.RED} Error while calling API: {e}")
        raise
def get(image, model="Salesforce/blip-image-captioning-base"):
    api_url = f"https://router.huggingface.co/hf-inference/models/{model}"

    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    buffered.seek(0)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(
        api_url,
        headers=headers,
        data=buffered.read()
    )

    if response.status_code != 200:
        return f"[HTTP {response.status_code} {response.reason}"

    try:
        result = response.json()
    except Exception:
        return "[Error] Response is not JSON"

    if isinstance(result, dict) and "error" in result:
        return f"[Error] {result['error']}"

    return result[0].get("generated_text", "No caption generated.")

def generate_text(prompt, model="NYTK/text-generation-news-gpt2-small-hungarian", max_new_tokens=60):
    print(f"{Fore.CYAN} Generating text with prompt: {prompt}")
    api_url=f"https://router.huggingface.co/hf-inference/models/{model}"
    payload={"inputs": prompt, "parameters":{"max_new_tokens":max_new_tokens}}
    text_bytes=query(api_url, payload=payload)
    try:
        result=json.loads(text_bytes.decode("utf-8"))
    except Exception as e:
        raise Exception("Failed to decode text generation prompt.")
    if isinstance(result,dict) and "error" in result:
        raise Exception(result['error'])
    generated=result[0].get("generated_text", "")
    return generated
def truncate(text, word_limit):
    words=text.strip().split()
    return " ".join(words[:word_limit])
def print_menu():
    print(f"""{Style.BRIGHT}
{Fore.GREEN}
Select Output Type:
1. Caption 
2. Description
3. Summary
4. Exit
""")
def main():
    image_path=input(f"{Fore.BLUE}Enter the path of the image: {Style.RESET_ALL}")
    if not os.path.exists(image_path):
        print(f"{Fore.RED}The file'{image_path}' does not exists.")
        return
    try:
        image=Image.open(image_path)
    except Exception as e:
        print(f"{Fore.RED}Failed to open image: {e}")
        return
    basic_caption=get(image)
    print(f"{Fore.YELLOW}Basic caption: {Style.BRIGHT}{basic_caption}\n")
    while True:
        print_menu()
        choice=input(f"{Fore.CYAN} Enter your choice (1-4): {Style.RESET_ALL}")
        if choice=='1':
            caption=truncate(basic_caption,5)
            print(f"{Fore.GREEN}Caption: {Style.BRIGHT}{caption}\n")
        elif choice=='2':
            prompt_text=f"Expand the following caption into a detailed description in exactly 30 words: {basic_caption}"
            try:
                generated=generate_text(prompt_text, max_new_tokens=40)
                description=truncate(generated,30)
                print(f"{Fore.GREEN}Description: {Style.BRIGHT}{description}\n")
            except Exception as e:
                print(f"{Fore.RED}Failed to generate summary: {e}")
        elif choice=='3':
            prompt_text=f"Expand the following caption into a detailed description in exactly 50 words: {basic_caption}"
            try:
                generated=generate_text(prompt_text, max_new_tokens=60)
                summary=truncate(generated,50)
                print(f"{Fore.GREEN}Description: {Style.BRIGHT}{summary}\n")
            except Exception as e:
                print(f"{Fore.RED}Failed to generate summary: {e}")
        elif choice=='4':
            print(f"{Fore.GREEN}Goodbye!")
            break
        else:
            print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 4.")
if __name__=="__main__":
    main()
