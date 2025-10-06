#!/usr/bin/env python3
"""
Analyseert 'latest' en 'previous' (voorlaatste) uit gallery/ en
schrijft outputs naar analysis/out/:

- latest-contact.png (256x256)
- previous-contact.png (256x256)
- compare-grid.png (256x256): 2x2 grid:
    TL: previous (128x128)
    TR: latest   (128x128)
    BL: abs-diff (grijs, 128x128)
    BR: edge-map (Sobel, 128x128)
- latest-hist.png (RGB-histogram)
- latest-report.txt / .json (statistieken latest)
- marker.json  (updated_at, source_latest, source_prev)
"""
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "gallery"
OUTDIR = ROOT / "analysis" / "out"
OUTDIR.mkdir(parents=True, exist_ok=True)

LATEST_JSON = GALLERY / "latest.json"

def load_latest_relpath():
    if not LATEST_JSON.exists():
        raise FileNotFoundError("gallery/latest.json niet gevonden")
    j = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    p = j.get("img") or j.get("file")
    if not p:
        raise ValueError("latest.json mist 'img' of 'file'")
    p = p.lstrip("/")  # normaliseer
    # als p bv "/ame-rasp-poe/gallery/01.webp" was, strip tot "gallery/01.webp"
    parts = p.split("/")
    if "gallery" in parts:
        p = "/".join(parts[parts.index("gallery"):])
    if not p.startswith("gallery/"):
        p = f"gallery/{p}"
    return p  # repo-relatief padstring

def find_previous_relpath(latest_rel):
    # Zoek vorige image in gallery/ op basis van bestandsnaam‐sort (lexicografisch).
    # Werkt voor namen 01.webp, 02.webp, piece-YYYYMMDDHHMM.webp, etc.
    exts = {".webp", ".png", ".jpg", ".jpeg"}
    files = sorted([p.name for p in GALLERY.iterdir()
                    if p.is_file() and p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError("Geen images in gallery/")
    latest_name = Path(latest_rel).name
    if latest_name not in files:
        # latest verwijst naar iets dat (nog) niet in de checkout staat
        # neem dan de op één na laatste
        if len(files) >= 2:
            return f"gallery/{files[-2]}"
        return f"gallery/{files[-1]}"
    idx = files.index(latest_name)
    if idx > 0:
        return f"gallery/{files[idx-1]}"
    # geen vorige, val terug op latest zelf
    return f"gallery/{files[idx]}"

def load_image_256(repo_rel):
    img = Image.open(ROOT / repo_rel).convert("RGB")
    # forceer 256x256 zonder blur
    return img.resize((256, 256), Image.NEAREST)

def save_contact(img, dest):
    # contact = gewoon 256x256
    img.save(dest, "PNG")

def save_histogram(img, dest):
    arr = np.array(img)
    plt.figure(figsize=(4, 2))
    plt.hist(arr[...,0].ravel(), bins=256, alpha=0.5, label="R")
    plt.hist(arr[...,1].ravel(), bins=256, alpha=0.5, label="G")
    plt.hist(arr[...,2].ravel(), bins=256, alpha=0.5, label="B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

def compute_metrics(img):
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean(axis=(0,1)).tolist()
    std  = arr.std(axis=(0,1)).tolist()
    gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=256, range=(0,255))
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p>0]*np.log2(p[p>0])).sum())
    return {
        "width": 256, "height": 256,
        "mean_rgb": {"r": mean[0], "g": mean[1], "b": mean[2]},
        "std_rgb":  {"r": std[0],  "g": std[1],  "b": std[2]},
        "entropy_gray_bits": entropy
    }

def sobel_edge(gray_u8):
    # eenvoudige Sobel edge magnitude
    g = gray_u8.astype(np.float32)
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    # 2D conv
    from scipy.signal import convolve2d
    gx = convolve2d(g, Kx, mode="same", boundary="symm")
    gy = convolve2d(g, Ky, mode="same", boundary="symm")
    mag = np.sqrt(gx*gx + gy*gy)
    mag = np.clip(mag / (mag.max()+1e-6) * 255, 0, 255).astype(np.uint8)
    return mag

def save_compare_grid(prev_img, latest_img, dest):
    # TL prev, TR latest, BL abs-diff (grijs), BR edge-map (Sobel op latest)
    prev_small   = prev_img.resize((128,128), Image.NEAREST)
    latest_small = latest_img.resize((128,128), Image.NEAREST)

    a = np.array(prev_small, dtype=np.int16)
    b = np.array(latest_small, dtype=np.int16)
    diff = np.abs(a - b).astype(np.uint8)
    diff_gray = (0.299*diff[...,0] + 0.587*diff[...,1] + 0.114*diff[...,2]).astype(np.uint8)
    diff_gray_rgb = np.stack([diff_gray]*3, axis=-1)

    latest_gray = (0.299*b[...,0] + 0.587*b[...,1] + 0.114*b[...,2]).astype(np.uint8)
    try:
        edges = sobel_edge(latest_gray)
    except Exception:
        # als scipy niet beschikbaar is, fallback op PIL EDGE_ENHANCE + normalisatie
        pil_edge = Image.fromarray(latest_small).filter(ImageFilter.FIND_EDGES).convert("L")
        edges = np.array(pil_edge)

    edges_rgb = np.stack([edges]*3, axis=-1)

    grid = Image.new("RGB", (256,256), "white")
    grid.paste(Image.fromarray(prev_small),   (0,0))
    grid.paste(Image.fromarray(latest_small), (128,0))
    grid.paste(Image.fromarray(diff_gray_rgb),(0,128))
    grid.paste(Image.fromarray(edges_rgb),    (128,128))
    grid.save(dest, "PNG")

def main():
    latest_rel = load_latest_relpath()
    prev_rel   = find_previous_relpath(latest_rel)

    latest_img = load_image_256(latest_rel)
    prev_img   = load_image_256(prev_rel)

    # contact sheets
    save_contact(latest_img, OUTDIR / "latest-contact.png")
    save_contact(prev_img,   OUTDIR / "previous-contact.png")

    # histogram + report voor latest
    save_histogram(latest_img, OUTDIR / "latest-hist.png")
    m = compute_metrics(latest_img)
    (OUTDIR / "latest-report.txt").write_text(
        "Latest image analysis report\n"
        "============================\n"
        f"Shape: 256x256 px\n"
        f"Mean RGB: ({m['mean_rgb']['r']:.2f}, {m['mean_rgb']['g']:.2f}, {m['mean_rgb']['b']:.2f})\n"
        f"Std  RGB: ({m['std_rgb']['r']:.2f}, {m['std_rgb']['g']:.2f}, {m['std_rgb']['b']:.2f})\n"
        f"Entropy (gray): {m['entropy_gray_bits']:.3f} bits\n",
        encoding="utf-8"
    )
    (OUTDIR / "latest-report.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

    # 256x256 vergelijkingsbeeld
    save_compare_grid(prev_img, latest_img, OUTDIR / "compare-grid.png")

    # marker (voor auto-refresh + client-side 'prev' resolutie)
    marker = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "source_latest": Path(latest_rel).name,
        "source_prev":   Path(prev_rel).name
    }
    (OUTDIR / "marker.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
