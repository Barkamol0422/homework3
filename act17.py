import requests
from config import api_key

model="nlpconnect/vit-gpt2-image-captioning"
api_url=f"https://api-inference.huggingface.co/models/{model}"
headers={"Authorization": f"Bearer {api_key}"}
def captions():
    image_source="man.png"
    try:
        with open(image_source, "rb") as f:
            image_bytes=f.read()
    except Exception as e:
        print(f"Error: {e}")
        return
    response=requests.post(api_url, headers=headers, data=image_bytes)
    result=response.json()
    if isinstance(result,dict) and "error" in result:
        print("Error")
    caption=result[0].get("generated_text", "No captions found")
    print("Image", image_source)
    print("Caption: ", caption)
def main():
    captions()
if __name__=="__main__":
    main()
