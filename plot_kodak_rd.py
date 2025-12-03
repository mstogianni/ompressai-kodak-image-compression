import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_results(csv_path: Path) -> pd.DataFrame:
    print("Reading results from:", csv_path)
    df = pd.read_csv(csv_path)

    # Περιμένουμε στήλες: img, model, quality, bpp, psnr, ms_ssim
    required = {"img", "model", "quality", "bpp", "psnr", "ms_ssim"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV is missing columns: {required - set(df.columns)}")

    return df


def make_rd_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Μέσος όρος ανά (model, quality)
    grouped = (
        df.groupby(["model", "quality"])
        .agg({"bpp": "mean", "psnr": "mean", "ms_ssim": "mean"})
        .reset_index()
    )

    models_in_order = [
        "bmshj2018-factorized",
        "bmshj2018-hyperprior",
        "mbt2018-mean",
        "cheng2020-attn",
    ]

    # PSNR–bpp
    plt.figure()
    for m in models_in_order:
        sub = grouped[grouped["model"] == m]
        if sub.empty:
            continue
        plt.plot(sub["bpp"], sub["psnr"], marker="o", label=m)

    plt.xlabel("bpp")
    plt.ylabel("PSNR (dB)")
    plt.title("Kodak RD curves (PSNR)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    psnr_path = out_dir / "kodak_rd_psnr.png"
    plt.savefig(psnr_path, dpi=200)
    plt.close()
    print("Saved:", psnr_path)

    # MS-SSIM–bpp
    plt.figure()
    for m in models_in_order:
        sub = grouped[grouped["model"] == m]
        if sub.empty:
            continue
        plt.plot(sub["bpp"], sub["ms_ssim"], marker="o", label=m)

    plt.xlabel("bpp")
    plt.ylabel("MS-SSIM")
    plt.title("Kodak RD curves (MS-SSIM)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    msssim_path = out_dir / "kodak_rd_msssim.png"
    plt.savefig(msssim_path, dpi=200)
    plt.close()
    print("Saved:", msssim_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="results/kodak_rd_results.csv",
        help="RD CSV from run_kodak_rd.py",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Folder to save plots",
    )
    args = parser.parse_args()

    wdir = Path(".").resolve()
    csv_path = (wdir / args.csv).resolve()
    out_dir = (wdir / args.outdir).resolve()

    df = load_results(csv_path)
    make_rd_plots(df, out_dir)


if __name__ == "__main__":
    main()
