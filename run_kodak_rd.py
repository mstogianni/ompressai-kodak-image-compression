import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    cheng2020_attn,
)


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)
    return x


def compute_metrics(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2)
    psnr = -10 * torch.log10(mse).item()
    ms_ssim_val = ms_ssim(x, x_hat, data_range=1.0, size_average=True).item()
    return psnr, ms_ssim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed_kodak",
        help="Φάκελος με τις ΗΔΗ επεξεργασμένες εικόνες (768 + center crop 512x512)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/kodak_rd_results.csv",
        help="CSV για αποθήκευση RD αποτελεσμάτων",
    )
    args = parser.parse_args()

    wdir = Path(".").resolve()
    input_dir = wdir / args.input
    output_csv = wdir / args.output

    print(f"Working dir: {wdir}")
    print(f"Loading images from: {input_dir}")
    print(f"Results CSV: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Μοντέλα
    model_fns = {
        "bmshj2018-factorized": bmshj2018_factorized,
        "bmshj2018-hyperprior": bmshj2018_hyperprior,
        "mbt2018-mean": mbt2018_mean,
        "cheng2020-attn": cheng2020_attn,
    }

    # Ladder ποιότητας (κοινό για 3 μοντέλα)
    main_qualities = [3, 4, 5, 6, 7]
    # Ablation: cheng2020-attn μόνο στα 3 και 4
    cheng_qualities = [3, 4]

    exts = [".png", ".jpg", ".jpeg"]
    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"[ERROR] No images found in {input_dir}")
        return

    print(f"Found {len(img_paths)} images")
    print(f"Models: {list(model_fns.keys())}")

    fieldnames = [
        "img",
        "model",
        "quality",
        "bpp",
        "psnr",
        "ms_ssim",
        "enc_time",
        "dec_time",
        "mem",
    ]

    # IMPORTANT: Κλείσε το CSV αν είναι ανοιχτό σε Excel, αλλιώς PermissionError
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        models_cache = {}

        for img_idx, img_path in enumerate(img_paths, 1):
            print(f"\nProcessing {img_path.name} ({img_idx}/{len(img_paths)})...")
            x = load_image(img_path, device)
            _, _, H, W = x.shape
            num_pixels = H * W

            for model_name, model_fn in model_fns.items():
                # Ablation: cheng μόνο στα 3,4 – τα άλλα full ladder
                if model_name == "cheng2020-attn":
                    qualities = cheng_qualities
                else:
                    qualities = main_qualities

                for q in qualities:
                    key = (model_name, q)

                    if key not in models_cache:
                        print(f"Loading {model_name} q={q}")
                        model = model_fn(quality=q, metric="mse", pretrained=True)
                        model = model.to(device).eval()
                        models_cache[key] = model
                    else:
                        model = models_cache[key]

                    with torch.no_grad():
                        # Encode
                        t0 = time.perf_counter()
                        out = model.compress(x)
                        enc_time = time.perf_counter() - t0

                        # bits / pixel
                        total_bits = 0
                        for s in out["strings"]:
                            total_bits += len(s[0]) * 8
                        bpp = total_bits / num_pixels

                        # Decode
                        t1 = time.perf_counter()
                        out_dec = model.decompress(out["strings"], out["shape"])
                        dec_time = time.perf_counter() - t1

                        # Διαφορετικές εκδόσεις CompressAI → robustness
                        if isinstance(out_dec, dict):
                            if "x_hat" in out_dec:
                                x_hat = out_dec["x_hat"]
                            elif "x" in out_dec:
                                x_hat = out_dec["x"]
                            else:
                                raise KeyError(
                                    f"Unknown keys in decompress output: {list(out_dec.keys())}"
                                )
                        else:
                            x_hat = out_dec

                        x_hat = x_hat.clamp_(0.0, 1.0)

                        psnr, ms_ssim_val = compute_metrics(x, x_hat)

                    row = {
                        "img": img_path.name,
                        "model": model_name,
                        "quality": q,
                        "bpp": bpp,
                        "psnr": psnr,
                        "ms_ssim": ms_ssim_val,
                        "enc_time": enc_time,
                        "dec_time": dec_time,
                        "mem": 0.0,  # placeholder
                    }
                    writer.writerow(row)

                    print(
                        f"{model_name} q={q} | "
                        f"bpp={bpp:.4f}, psnr={psnr:.2f}, ms-ssim={ms_ssim_val:.5f}, "
                        f"enc_time={enc_time:.3f}s, dec_time={dec_time:.3f}s"
                    )

    print("\nAll done! CSV saved to:", output_csv)


if __name__ == "__main__":
    main()
