import argparse
import base64
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests


# ---------------------------
# Readiness gate
# ---------------------------

def wait_for_ollama_ready(ollama_generate_url: str, timeout_s: int = 180) -> None:
    # Turn ".../api/generate" into ".../api/version"
    base = ollama_generate_url.split("/api/")[0].rstrip("/")
    version_url = f"{base}/api/version"

    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(version_url, timeout=5)
            if r.status_code == 200:
                return
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(2)

    raise RuntimeError(f"Ollama not ready after {timeout_s}s. Last error: {last_err}")


# ---------------------------
# Image utilities (mirrors your glm_ocr_local style)
# ---------------------------

def enhance_gray(img_bgr, target_long: int = 2600):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    long_side = max(h, w)

    if target_long and long_side > target_long:
        scale = target_long / float(long_side)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    bg = cv2.medianBlur(gray, 31)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    blur = cv2.GaussianBlur(norm, (0, 0), 1.1)
    sharp = cv2.addWeighted(norm, 1.35, blur, -0.35, 0)
    return sharp


def deskew(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bw
    coords = np.column_stack(np.where(inv > 0))
    if coords.size < 2000:
        return gray

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.7:
        return gray

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def crop_foreground(gray, margin: int = 60):
    inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 12
    )
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        h, w = gray.shape
        return gray, (0, 0, w, h)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w = gray.shape

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w - 1, x1 + margin)
    y1 = min(h - 1, y1 + margin)

    crop = gray[y0:y1 + 1, x0:x1 + 1]
    return crop, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def resize_long_side(img, target_long: int, upscale_limit: float = 1.0):
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side == 0:
        return img

    scale = target_long / float(long_side)
    if scale > 1.0:
        scale = min(scale, upscale_limit)
    if abs(scale - 1.0) < 1e-3:
        return img

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def split_vertical_with_overlap(img, parts: int = 2, overlap_px: int = 200):
    h, w = img.shape[:2]
    if parts <= 1 or h < 900:
        return [img]

    step = h // parts
    out = []
    for i in range(parts):
        y0 = max(0, i * step - (overlap_px if i > 0 else 0))
        y1 = min(h, (i + 1) * step + (overlap_px if i < parts - 1 else 0))
        out.append(img[y0:y1, :])
    return out


# ---------------------------
# Text cleanup
# ---------------------------

_FENCE_LINE = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*$")


def cleanup_ocr_text(s: str) -> str:
    if not s:
        return ""
    lines = []
    for line in s.splitlines():
        if _FENCE_LINE.match(line):
            continue
        lines.append(line.rstrip())
    out = "\n".join(lines).strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def is_effectively_empty(s: str) -> bool:
    return len(cleanup_ocr_text(s or "").strip()) < 5


def should_skip_existing(out_file: Path) -> bool:
    try:
        if not out_file.exists() or out_file.stat().st_size == 0:
            return False
        head = out_file.read_text(encoding="utf-8", errors="ignore")[:2000]
        return not is_effectively_empty(head)
    except Exception:
        return False


# ---------------------------
# Ollama helpers
# ---------------------------

def _b64_png_from_bgr(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_text_from_ollama_json(data: dict) -> str:
    # If Ollama returns an explicit error, surface it
    err = data.get("error")
    if isinstance(err, str) and err.strip():
        raise RuntimeError(err.strip())

    # /api/generate shape
    if isinstance(data.get("response"), str):
        return data["response"]

    # /api/chat shape (sometimes returned by some setups)
    msg = data.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]

    # other fallbacks
    for k in ("output", "text", "content"):
        v = data.get(k)
        if isinstance(v, str):
            return v

    return ""


def ollama_ocr_one_image(
    ollama_generate_url: str,
    model: str,
    image_b64: str,
    prompt: str,
    timeout_s: int,
    num_predict: int,
    debug_json_path: Optional[Path] = None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        },
    }

    r = requests.post(ollama_generate_url, json=payload, timeout=timeout_s)
    # Keep body for debugging (Ollama often returns a JSON error)
    if r.status_code != 200:
        body = (r.text or "")[:2000]
        raise RuntimeError(f"Ollama HTTP {r.status_code} from {ollama_generate_url}. Body: {body}")


    raw_body = r.text
    try:
        data = r.json()
    except Exception:
        # Save raw response if it's not JSON
        if debug_json_path:
            debug_json_path.write_text(raw_body, encoding="utf-8")
        return ""

    if debug_json_path:
        debug_json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    out = _extract_text_from_ollama_json(data)
    return cleanup_ocr_text(str(data.get("response", "")).strip())


# ---------------------------
# OCR pipeline
# ---------------------------

def ocr_page(
    image_path: Path,
    model: str,
    ollama_generate_url: str,
    prompt: str,
    preprocess_long: int = 2600,
    min_long_for_ocr: int = 1800,
    tile: bool = True,
    max_new_tokens: int = 1200,
    debug_dir: Optional[Path] = None,
) -> str:
    img = cv2.imread(str(image_path))
    if img is None:
        return ""

    gray = enhance_gray(img, target_long=preprocess_long)
    gray = deskew(gray)
    crop, _ = crop_foreground(gray)
    crop = resize_long_side(crop, target_long=min_long_for_ocr, upscale_limit=1.0)

    bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    slices = split_vertical_with_overlap(bgr, parts=2, overlap_px=200) if tile else [bgr]

    texts: List[str] = []
    for si, sl in enumerate(slices):
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{image_path.stem}__prep_{si}.png"), sl)

        image_b64 = _b64_png_from_bgr(sl)

        json_path = None
        if debug_dir:
            json_path = debug_dir / f"{image_path.stem}__resp_{si}.json"

        txt = ollama_ocr_one_image(
            ollama_generate_url=ollama_generate_url,
            model=model,
            image_b64=image_b64,
            prompt=prompt,
            timeout_s=240,
            num_predict=max_new_tokens,
            debug_json_path=json_path,
        )

        if debug_dir:
            (debug_dir / f"{image_path.stem}__raw_{si}.txt").write_text(txt, encoding="utf-8")

        if not is_effectively_empty(txt):
            texts.append(txt)

    # fallback if tiling produced nothing
    if not texts and tile:
        image_b64 = _b64_png_from_bgr(bgr)

        json_path = None
        if debug_dir:
            json_path = debug_dir / f"{image_path.stem}__resp_full.json"

        txt = ollama_ocr_one_image(
            ollama_generate_url=ollama_generate_url,
            model=model,
            image_b64=image_b64,
            prompt=prompt,
            timeout_s=240,
            num_predict=max_new_tokens,
            debug_json_path=json_path,
        )

        if debug_dir:
            (debug_dir / f"{image_path.stem}__raw_full.txt").write_text(txt, encoding="utf-8")

        if not is_effectively_empty(txt):
            texts.append(txt)

    final = "\n\n".join(texts).strip()

    # Never silently empty: emit sentinel if nothing readable
    if is_effectively_empty(final):
        return "[OCR_EMPTY]"

    return final


def run_folder(
    input_dir: str,
    out_dir: str,
    model: str,
    ollama_generate_url: str,
    resume: bool = True,
    debug: bool = True,
    tile: bool = True,
    max_new_tokens: int = 1200,
    prompt: str = (
        "You are an OCR engine. Transcribe ALL visible text from the image.\n"
        "Rules:\n"
        "- Output ONLY the text (no markdown, no code fences).\n"
        "- Preserve line breaks as best as possible.\n"
        "- Do not add commentary or explanations.\n"
        "- If you cannot read any text, output exactly: [OCR_EMPTY]\n"
    ),
):
    # Ensure Ollama is actually ready before processing
    wait_for_ollama_ready(ollama_generate_url, timeout_s=240)

    input_path = Path(input_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    debug_dir = out_path / "_debug" if debug else None
    log_file = out_path / "run_status.log"

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])

    header = (
        f"=== Ollama OCR run ===\n"
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"in_dir: {input_dir}\n"
        f"out_dir: {out_dir}\n"
        f"ollama_generate_url: {ollama_generate_url}\n"
        f"model: {model}\n"
        f"tile: {tile}\n"
        f"max_new_tokens: {max_new_tokens}\n"
        f"num_images: {len(images)}\n"
        f"======================\n"
    )
    print(header)
    log_file.write_text(header, encoding="utf-8")

    t0 = time.time()
    for idx, img_file in enumerate(images, start=1):
        out_file = out_path / f"{img_file.stem}.txt"

        if resume and should_skip_existing(out_file):
            msg = f"[SKIP] {idx:03d}/{len(images):03d} {img_file.name} (already done)"
            print(msg)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            continue

        start = time.time()
        try:
            text = ocr_page(
                image_path=img_file,
                model=model,
                ollama_generate_url=ollama_generate_url,
                prompt=prompt,
                tile=tile,
                max_new_tokens=max_new_tokens,
                debug_dir=debug_dir,
            )
            out_file.write_text(text, encoding="utf-8")

            dt = time.time() - start
            msg = f"[OK ] {idx:03d}/{len(images):03d} {img_file.name} ({dt:.1f}s) chars={len(text)}"
            print(msg)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

        except Exception as e:
            dt = time.time() - start
            msg = f"[FAIL] {idx:03d}/{len(images):03d} {img_file.name} ({dt:.1f}s) {e}"
            print(msg)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"{img_file.stem}__error.txt").write_text(str(e), encoding="utf-8")
            out_file.write_text("", encoding="utf-8")

    done_msg = f"\nDone in {(time.time() - t0) / 60:.1f} minutes.\n"
    print(done_msg)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(done_msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/data/input_images", help="Folder of page images")
    ap.add_argument("--out_dir", default="/data/output_text", help="Where to write .txt outputs")
    ap.add_argument("--ollama_url", default="http://ollama:11434/api/generate", help="Ollama URL (/api/generate)")
    ap.add_argument("--model", required=True, help="Ollama OCR model name (e.g., glm-ocr:latest)")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--no_debug", action="store_true")
    ap.add_argument("--no_tile", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--prompt", default=None, help="Override OCR prompt")
    args = ap.parse_args()

    prompt = args.prompt or (
        "You are an OCR engine. Transcribe ALL visible text from the image.\n"
        "Rules:\n"
        "- Output ONLY the text (no markdown, no code fences).\n"
        "- Preserve line breaks as best as possible.\n"
        "- Do not add commentary or explanations.\n"
        "- If you cannot read any text, output exactly: [OCR_EMPTY]\n"
    )

    run_folder(
        input_dir=args.in_dir,
        out_dir=args.out_dir,
        model=args.model,
        ollama_generate_url=args.ollama_url,
        resume=not args.no_resume,
        debug=not args.no_debug,
        tile=not args.no_tile,
        max_new_tokens=args.max_new_tokens,
        prompt=prompt,
    )


if __name__ == "__main__":
    main()
