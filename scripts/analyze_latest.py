import json, base64, io, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
GALLERY = REPO_ROOT / "gallery"
OUTDIR  = REPO_ROOT / "analysis" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

LATEST_JSON = GALLERY / "latest.json"

def load_latest_path():
    if LATEST_JSON.exists():
        j = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
        if "img" in j and j["img"]:
            p = j["img"]
        elif "file" in j and j["file"]:
            p = j["file"]
        else:
            raise SystemExit("latest.json mist 'img' of 'file'")
        # normaliseer pad: accepteer '01.webp', 'gallery/01.webp', '/ame-.../gallery/01.webp'
        p = p.strip()
        if p.startswith("http"):
            raise SystemExit("Geef een repo-pad in latest.json, geen absolute URL.")
        if p.startswith("/"):
            # haal repo-root prefix eruit indien aanwezig
            parts = p.split("/")
            # zoek 'gallery' segment
            if "gallery" in parts:
                i = parts.index("gallery")
                p = "/".join(parts[i:])  # vanaf gallery/...
            else:
                raise SystemExit(f"Onverwacht pad in latest.json: {p}")
        if not p.startswith("gallery/"):
            p = f"gallery/{p}"
        return REPO_ROOT / p
    else:
        raise SystemExit("gallery/latest.json niet gevonden")

def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"Bestand niet gevonden: {path}")
    im = Image.open(path).convert("RGB")
    # forceer 256x256 canvas voor analyse (resample: nearest om artefacten te vermijden)
    im = im.resize((256, 256), Image.NEAREST)
    return im

def compute_metrics(arr: np.ndarray) -> dict:
    # arr shape (H,W,3), uint8
    H, W, _ = arr.shape
    total = H * W
    ch = [arr[...,i].astype(np.float32) for i in range(3)]

    means = [float(c.mean()) for c in ch]
    stds  = [float(c.std())  for c in ch]
    # eenvoudige grijswaarden-histogram
    gray = (0.299*ch[0] + 0.587*ch[1] + 0.114*ch[2]).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0,255))
    hist = hist.astype(int).tolist()
    # Shannon entropy in bits (van grijswaarden)
    p = np.array(hist, dtype=np.float64)
    p = p / max(p.sum(), 1)
    entropy = -np.sum(p * np.log2(np.where(p>0, p, 1))).item()

    return {
        "width": W, "height": H, "pixels": int(total),
        "mean_rgb": {"r": means[0], "g": means[1], "b": means[2]},
        "std_rgb":  {"r": stds[0],  "g": stds[1],  "b": stds[2]},
        "entropy_gray_bits": float(entropy)
    }

def save_histogram(gray: np.ndarray, out_png: Path):
    plt.figure(figsize=(4,1.5), dpi=128)
    plt.hist(gray.ravel(), bins=256, range=(0,255))
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def save_contact_sheet(im: Image.Image, out_png: Path):
    # maak een kleine triptiek: origineel + verhelderd + posterize
    arr = np.array(im)
    # verhelder
    bright = np.clip(arr + 30, 0, 255).astype(np.uint8)
    # posterize 6 levels
    step = 255/5
    post = (np.round(arr/step)*step).clip(0,255).astype(np.uint8)

    sheet = Image.new("RGB", (256*3, 256), "white")
    sheet.paste(Image.fromarray(arr), (0,0))
    sheet.paste(Image.fromarray(bright), (256,0))
    sheet.paste(Image.fromarray(post), (512,0))
    sheet.save(out_png, format="PNG")

def main():
    img_path = load_latest_path()
    im = load_image(img_path)
    arr = np.array(im)
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8)

    metrics = compute_metrics(arr)
    (OUTDIR / "latest-report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_histogram(gray, OUTDIR / "latest-hist.png")
    save_contact_sheet(im, OUTDIR / "latest-contact.png")
    # plain text samenvatting
    summary = [
        f"Size: {metrics['width']}x{metrics['height']} px",
        f"Mean RGB: ({metrics['mean_rgb']['r']:.1f}, {metrics['mean_rgb']['g']:.1f}, {metrics['mean_rgb']['b']:.1f})",
        f"Std  RGB: ({metrics['std_rgb']['r']:.1f}, {metrics['std_rgb']['g']:.1f}, {metrics['std_rgb']['b']:.1f})",
        f"Entropy (gray): {metrics['entropy_gray_bits']:.3f} bits"
    ]
    (OUTDIR / "latest-report.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
