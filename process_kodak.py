import argparse
from pathlib import Path

from PIL import Image


def process_image(img_path, dst_dir, target_long_side=768, crop_size=512):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # 1) Scale ώστε η ΜΕΓΑΛΥΤΕΡΗ πλευρά να γίνει target_long_side
    long_side = max(w, h)
    scale = target_long_side / long_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 2) Center crop σε crop_size x crop_size (512 x 512)
    left = max(0, (new_w - crop_size) // 2)
    top = max(0, (new_h - crop_size) // 2)
    right = left + crop_size
    bottom = top + crop_size
    img = img.crop((left, top, right, bottom))

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / img_path.name
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        default="data/kodak",
        help="Φάκελος με τις αρχικές Kodak εικόνες",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/processed_kodak",
        help="Φάκελος για τις επεξεργασμένες (768 + center crop 512x512)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"[ERROR] Δεν βρέθηκε ο φάκελος πηγής: {src_dir}")
        return

    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    img_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)

    if not img_paths:
        print(f"[ERROR] Δεν βρέθηκαν εικόνες στο {src_dir}")
        return

    print(f"Source dir: {src_dir}")
    print(f"Dest dir:   {dst_dir}")
    print(f"Found {len(img_paths)} images")

    for i, img_path in enumerate(img_paths, 1):
        out_path = process_image(img_path, dst_dir)
        print(f"[{i}/{len(img_paths)}] {img_path.name} -> {out_path.name}")

    print("Done. Processed images saved to:", dst_dir)


if __name__ == "__main__":
    main()
