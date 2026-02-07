import requests
import base64
from PIL import Image
from io import BytesIO
from config import api_key

def generate_image(prompt, image_path, mask_path):
    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-inpainting"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "image/png"
    }

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    with open(mask_path, "rb") as f:
        base64_mask = base64.b64encode(f.read()).decode()

    payload = {
        "inputs": {
            "prompt": prompt,
            "image": base64_image,
            "mask": base64_mask
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Request failed: {response.status_code} {response.text}")

def main():
    print("Welcome to inpainting and restoration challenge!")
    while True:
        prompt = input("Enter a prompt: ")
        if prompt.lower() == "exit":
            print("Goodbye")
            break

        image_path = input("Enter path to base image: ")
        if image_path.lower() == "exit":
            print("Goodbye")
            break

        mask_path = input("Enter the path to the mask image: ")
        if mask_path.lower() == "exit":
            break

        try:
            print("Processing image")
            result_image = generate_image(prompt, image_path, mask_path)
            result_image.show()
            save_option = input("Do you want to save the inpainted image: ").strip().lower()
            if save_option == "yes":
                file_name = input("Enter a name for the picture: ")
                result_image.save(f"{file_name}.png")
                print(f"Image saved as {file_name}.png")
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
