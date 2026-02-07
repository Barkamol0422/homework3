import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from config import api_key

def generate_image(prompt):
    api_url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3-medium-diffusers"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt}

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(response.text)

def post_process_image(image):
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = image.filter(ImageFilter.SHARPEN)
    return image

prompt = input("Enter prompt: ")
image = generate_image(prompt)
image = post_process_image(image)
image.show()

choice = input("Save image? (y/n): ").lower()
if choice == "y":
    filename = input("Enter file name (with .png or .jpg): ")
    image.save(filename)
