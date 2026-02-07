import requests
from config import api_key
from colorama import Fore, Style, init

init(autoreset=True)

default_model = "google/pegasus-xsum"


def build_api_url(model_name):
    return f"https://router.huggingface.co/hf-inference/models/{model_name}"


def query(payload, model_name=default_model):
    api_url = build_api_url(model_name)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


def summarize_text(text, min_length, max_length, model_name=default_model):
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length,
            "do_sample": False
        }
    }

    print(Fore.BLUE + Style.BRIGHT +
          f"\n🚀 Performing AI summarization using model: {model_name}")

    result = query(payload, model_name=model_name)

    if isinstance(result, list) and result and "summary_text" in result[0]:
        return result[0]["summary_text"]
    else:
        print(Fore.RED + "❌ Error in summarization response:")
        print(result)
        return None


if __name__ == "__main__":
    print(Fore.YELLOW + Style.BRIGHT + "👋 Hi there! What's your name:")
    user_name = input("Your name: ").strip() or "User"

    print(Fore.GREEN + f"Welcome, {user_name}! Let's give your text some AI magic ✨")

    print(Fore.YELLOW + Style.BRIGHT + "\nPlease enter text you want to summarize:")
    user_text = input("> ").strip()

    if not user_text:
        print(Fore.RED + "No text provided. Exiting.")
        exit()

    print(Fore.YELLOW + "\nEnter the model name you want to use:")
    model_choice = input("Model name: ").strip() or default_model

    print(Fore.YELLOW + "\nChoose your summarization style:")
    print("1. Standard Summary")
    print("2. Enhanced Summary")

    style_choice = input("Enter 1 or 2: ").strip()

    if style_choice == "2":
        min_length, max_length = 80, 200
        print(Fore.BLUE + "Enhancing summarization process... ")
    else:
        min_length, max_length = 50, 150
        print(Fore.BLUE + "Using standard summarization settings... ")

    summary = summarize_text(user_text, min_length, max_length, model_choice)

    if summary:
        print(Fore.GREEN + Style.BRIGHT +
              f"\nAI summarizer output for {user_name}:")
        print(Fore.GREEN + summary)
    else:
        print(Fore.RED + "Failed to generate summary.")
