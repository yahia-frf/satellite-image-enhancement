# 🌤️ Satellite Image Enhancement & Cloud Removal using GANs

This project implements an **end-to-end PyTorch pipeline** for **satellite image classification and enhancement** on the **EuroSAT dataset**.  
It explores both **ResNet-based classifiers** and **Pix2Pix / CycleGAN architectures** for **cloud removal** and **scene restoration**, achieving strong quantitative and qualitative results.

---

## 🚀 Project Overview

### 🎯 Objective
Enhance and reconstruct satellite images affected by cloud cover, enabling clearer Earth observation data for downstream applications such as land classification, vegetation monitoring, and environmental analysis.

### ⚙️ Key Features
- **EuroSAT Dataset Integration** (RGB images of land use classes)
- **ResNet-18 Classifier** trained on 10 classes with 95%+ accuracy  
- **GAN-based Enhancement Models** (Pix2Pix & CycleGAN) for cloud removal  
- **Before/After Visualization Dashboard** with automated HTML report generation  
- **SSIM / PSNR Evaluation** for reconstruction quality  
- **Virtual Environment + Requirements** for full reproducibility  

---

## 🧠 Model Architecture

### 🟢 Classification (EuroSAT)
- Backbone: `ResNet-18` pretrained on ImageNet  
- Output: 10 land-use classes  
- Optimizer: Adam (`lr=1e-4`)  
- Loss: CrossEntropy  
- Validation Accuracy: **95.33%**

### ☁️ Cloud Removal (GANs)
- **Pix2Pix** with:
  - Generator: U-Net (skip connections)
  - Discriminator: PatchGAN
- Trained on synthetic cloudy–clear image pairs  
- Achieved **SSIM ≈ 0.89**, demonstrating strong visual fidelity

---

## 🧩 Project Structure

📦 satellite-image-enhancement
├── src/
│ ├── dataset.py # EuroSAT dataset loader
│ ├── model.py # ResNet model definition
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation + visualization
├── results/ # Saved outputs and visual comparisons
├── data/ # Dataset directory (EuroSAT)
├── main.py # Entry point for training & evaluation
├── notebook.ipynb # Interactive analysis / visualization
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🧰 Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yahia-frf/satellite-image-enhancement.git
cd satellite-image-enhancement

# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate   # on Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt
