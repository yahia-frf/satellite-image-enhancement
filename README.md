# 🌤️ Satellite Image Enhancement & Cloud Removal using GANs

> 🚀 An end-to-end PyTorch project for enhancing and restoring satellite images using **GANs** and **ResNet-based classifiers** on the **EuroSAT dataset**.

---

## 🛰️ Overview

This project focuses on **restoring and enhancing satellite images obscured by cloud cover**, improving the usability of Earth observation data for applications like:
- 🌾 **Land use classification**
- 🌳 **Vegetation and environmental monitoring**
- 🏙️ **Urban area detection**

The system integrates **deep learning** techniques (ResNet & GANs) to:
- Classify satellite images into land-use categories.
- Remove or reduce cloud artifacts using **Pix2Pix** and **CycleGAN** architectures.

---

## ⚙️ Features

✅ **EuroSAT Dataset Integration** — RGB satellite imagery covering 10 land-use classes  
✅ **ResNet-18 Classifier** — Achieves over **95% accuracy** on test data  
✅ **Pix2Pix / CycleGAN** — Used for cloud removal and image restoration  
✅ **SSIM & PSNR Metrics** — Evaluate reconstruction quality  
✅ **Visualization Dashboard** — Compare before/after restoration  
✅ **Reproducible Setup** — Full environment reproducibility with `venv` and `requirements.txt`  

---

## 🧠 Model Architectures

### 🟢 ResNet-18 Classifier
- Pretrained on ImageNet  
- Fine-tuned on EuroSAT (10 classes)  
- **Optimizer:** Adam (`lr=1e-4`)  
- **Loss:** CrossEntropy  
- **Validation Accuracy:** ~95.3%  

### ☁️ GANs for Cloud Removal
- **Pix2Pix**:
  - Generator: U-Net (skip connections)
  - Discriminator: PatchGAN
- **CycleGAN**:
  - Two generators + two discriminators for unsupervised translation
- **Metrics:**
  - SSIM ≈ 0.89  
  - PSNR ≈ 27.5 dB  
- **Result:** Visually realistic restoration with preserved texture and color

---

## 🧩 Project Structure

