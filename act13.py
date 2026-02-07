import requests
from PIL import Image
from io import BytesIO
from config import api_key

api_url="https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-3-medium-diffusers"

def generate_image(prompt: str) -> Image.Image:
    headers={"Authorization": f"Bearer {api_key}"}
    payload={"inputs": prompt}
    try:
        response=requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        if 'image' in response.headers.get('Content-Type',''):
            image=Image.open(BytesIO(response.content))
            return image
        else:
            raise Exception("Response is not a image. It might be an error message.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")
def main():
    print("Welcome to text to imag egeneration")
    print("Type exit to exit")
    while True:
        prompt=input("Enter a description for the image: ")
        if prompt.lower()=="exit":
            print("Goodbye!")
            exit()
        print("\nGenerating image...\n")
        try:
            image=generate_image(prompt)
            image.show()
            save_option=input("Do you want to save the image?: ").strip().lower()
            if save_option=="yes":
                file_name=input("Enter a name for the image file: ").strip() or "generated_image"
                image.save(f"{file_name}.png")
                print(f"Image saved as {file_name}.png\n")
        except Exception as e:
            print(f"An error occured: {e}\n")
        print("-"*80+"\n")
if __name__=="__main__":
    main()
                
        
