from pathlib import Path
from PIL import Image

def resize_to_short_side(img: Image.Image, short_side: int, resample) -> Image.Image:
    w, h = img.size
    if min(w, h) == short_side:
        return img
    if w < h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side
        new_w = int(w * short_side / h)
    return img.resize((new_w, new_h), resample=resample)

def process_split(img_dir: Path, mask_dir: Path, out_img_dir: Path, out_mask_dir: Path, short_side: int):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in img_dir.glob("*.jpg")])
    for img_path in image_files:
        name = img_path.stem
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            continue

        out_img_path = out_img_dir / f"{name}.jpg"
        out_mask_path = out_mask_dir / f"{name}.png"

        if out_img_path.exists() and out_mask_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_r = resize_to_short_side(img, short_side, Image.BILINEAR)
        mask_r = resize_to_short_side(mask, short_side, Image.NEAREST)

        img_r.save(out_img_path, quality=95)
        mask_r.save(out_mask_path)