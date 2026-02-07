import requests
import os, io, time, random, mimetypes
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from config import api_key

model = "facebook/detr-resnet-50"
api = f"https://router.huggingface.co/hf-inference/models/{model}"

allowed = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
max_mb = 8

emoji = {
    "person": "🧍",
    "man": "👨",
    "woman": "👩",
    "child": "🧒",
    "car": "🚗",
    "bus": "🚌",
    "truck": "🚚",
    "bicycle": "🚲",
    "motorcycle": "🏍️",
    "airplane": "✈️",
    "dog": "🐶",
    "cat": "🐱",
    "horse": "🐎",
    "bird": "🐦",
    "cow": "🐄",
    "chair": "🪑",
    "sofa": "🛋️",
    "table": "🪟",
    "laptop": "💻",
    "tv": "📺",
    "cell phone": "📱",
    "book": "📘",
    "cup": "☕",
    "bottle": "🍼",
    "backpack": "🎒"
}

def font(sz=18):
    for f in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except:
            pass
    return ImageFont.load_default()

def ask_image():
    print("\nPick an image:")
    while True:
        p = input("Image path: ").strip().strip('"').strip("'")
        if not p or not os.path.isfile(p):
            print("Not found")
            continue
        if os.path.splitext(p)[1].lower() not in allowed:
            print("Unsupported type")
            continue
        if os.path.getsize(p) / (1024 * 1024) > max_mb:
            print("Too big")
            continue
        try:
            Image.open(p).verify()
        except:
            print("Corrupted image")
            continue
        return p

def infer(path, image_bytes, tries=8):
    mime, _ = mimetypes.guess_type(path)
    for _ in range(tries):
        if mime and mime.startswith("image/"):
            r = requests.post(
                api,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": mime
                },
                data=image_bytes,
                timeout=60
            )
        else:
            r = requests.post(
                api,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"inputs": (os.path.basename(path), image_bytes)},
                timeout=60
            )
        if r.status_code == 200:
            d = r.json()
            if isinstance(d, dict) and "error" in d:
                raise RuntimeError(d["error"])
            if not isinstance(d, list):
                raise RuntimeError("Bad API response")
            return d
        if r.status_code == 503:
            time.sleep(2)
            continue
        raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Model warm-up timeout")

def draw(img, dets, thr=0.5):
    d = ImageDraw.Draw(img)
    f = font(18)
    counts = {}
    for det in dets[:50]:
        s = float(det.get("score", 0))
        if s < thr:
            continue
        lab = det.get("label", "object")
        b = det.get("box", {})
        x1, y1, x2, y2 = (int(b.get(k, 0)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if not (x2 > 0 and y2 > 0):
            x = int(b.get("x", 0))
            y = int(b.get("y", 0))
            w = int(b.get("w", 0))
            h = int(b.get("h", 0))
            x1, y1, x2, y2 = x, y, x + w, y + h
        color = tuple(random.randint(80, 255) for _ in range(3))
        d.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
        txt = f"{emoji.get(lab.lower(), '')} {lab} {s*100:.0f}%"
        tw = d.textlength(txt, font=f)
        th = f.size + 6
        d.rectangle([(x1, max(0, y1 - th)), (x1 + tw + 8, y1)], fill=color)
        d.text((x1 + 4, y1 - th + 3), txt, font=f, fill=(0, 0, 0))
        counts[lab] = counts.get(lab, 0) + 1
    return counts

def main():
    path = ask_image()
    with open(path, "rb") as fh:
        by = fh.read()
    try:
        dets = infer(path, by)
    except Exception as e:
        print("Error:", e)
        return
    img = Image.open(io.BytesIO(by)).convert("RGB")
    counts = draw(img, dets, thr=0.5)
    a = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"annotated_{a}.png"
    img.save(out)
    print(f"Saved as: {out}")
    if counts:
        print("I found:")
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{emoji.get(k.lower(), '')} {k}: {v}")
    else:
        print("No confident detections")
    print("\nDisclaimer. This is an AI model demo.")

if __name__ == "__main__":
    main()
