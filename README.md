# Image Compression Experiments with CompressAI (Kodak Dataset)

This repository contains a small experimental pipeline for **learned image compression** using [CompressAI](https://github.com/InterDigitalInc/CompressAI) on the **Kodak** image dataset.

It includes:

- preprocessing and cropping of the images  
- running several neural compression models at different quality levels  
- logging **rate–distortion (RD)** metrics (bpp, PSNR, MS-SSIM)  
- plotting RD curves for model comparison  

---

## 📁 Project Structure

- `process_kodak.py`  
  Preprocesses the original Kodak images:
  - resizes each image so that the **longest side = 768 px**
  - applies a **center crop of 512×512**
  - saves the processed images into a target folder

- `run_kodak_rd.py`  
  Runs multiple CompressAI models over the processed images and computes RD metrics:

  - Models used (via `compressai.zoo`):
    - `bmshj2018-factorized`
    - `bmshj2018-hyperprior`
    - `mbt2018-mean`
    - `cheng2020-attn`
  - For a range of **quality levels** (e.g. 1–8)
  - For each image and model:
    - compresses and reconstructs the image
    - computes **bpp** (bits per pixel)
    - computes **PSNR**
    - computes **MS-SSIM** using `pytorch_msssim`
  - Stores all results in a CSV file.

- `plot_kodak_rd.py`  
  Loads the CSV results and:
  - groups by **(model, quality)**  
  - averages `bpp`, `psnr`, `ms_ssim`
  - produces **rate–distortion plots**, e.g. PSNR vs bpp for each model

---
## 🔬 Ablation Experiment (Cheng2020-attn)

The `cheng2020-attn` model was **not evaluated across the full quality range**.  
Instead, it was included **only as an ablation model** to examine how architectural
changes—specifically attention modules—affect compression behavior.

Unlike the baseline models (`bmshj2018-factorized`, `bmshj2018-hyperprior`, `mbt2018-mean`),
Cheng2020-attn was used **exclusively to observe the effect of the attention mechanism**,  
not to generate full RD curves.

This ablation highlights:

- the impact of attention layers on reconstruction quality  
- architectural differences rather than quality-level differences  
- how network design choices influence RD performance independently of bitrate settings  
---

## 🔧 Dependencies

- Python 3.x
- PyTorch
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- `pytorch-msssim`
- `Pillow`
- `pandas`
- `matplotlib`
- `torchvision`

Install them (example):

```bash
pip install compressai pytorch-msssim pillow pandas matplotlib torchvision
```
Make sure PyTorch is installed with CPU or GPU support, depending on your setup.


▶️ Usage
1️⃣ Preprocess Kodak images
Assume you have the original Kodak images in data/kodak/.

Run:
```bash
python process_kodak.py --input data/kodak --output data/processed_kodak
```
This will:

load each image

resize it (longest side = 768)

center-crop to 512×512

save to data/processed_kodak/

2️⃣ Run compression experiments and collect RD data
```bash
python run_kodak_rd.py \
  --input data/processed_kodak \
  --output results/kodak_rd_results.csv
```
This script:

loops over all images in data/processed_kodak/

for each model & quality:

compresses and reconstructs the image

computes bpp, PSNR, MS-SSIM

writes everything to results/kodak_rd_results.csv

3️⃣ Plot RD curves
```bash
python plot_kodak_rd.py \
  --csv results/kodak_rd_results.csv \
  --output plots/
```
This will generate RD curves for the different models and store them in the plots/ directory.

📊 Output
You will obtain:

A CSV file with columns like:

img

model

quality

bpp

psnr

ms_ssim

Plots that compare:

PSNR vs bpp per model

optionally MS-SSIM vs bpp per model

These can be used to analyze how different neural compression models behave on the Kodak dataset across different bitrates.

🎓 What This Project Demonstrates
Practical use of CompressAI for learned image compression

Full experimental pipeline: preprocessing → evaluation → visualization

Computation of standard rate–distortion metrics (bpp, PSNR, MS-SSIM)

Basic research-style workflow in Python (scripts + CSV + plots)

This repository is a compact example of how to build an experimental framework for image compression research.
