# 🌤️ Satellite Image Enhancement & Cloud Removal using GANs

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/yahia-frf/satellite-image-enhancement?style=social)](https://github.com/yahia-frf/satellite-image-enhancement/stargazers)

**An end-to-end deep learning solution for enhancing and restoring satellite images using GANs and ResNet-based classifiers on the EuroSAT dataset.**

[Features](#-features) • [Installation](#-installation--setup) • [Usage](#-usage) • [Results](#-results-summary) • [Contributing](#-contributing)

</div>

---

## 🛰️ Overview

This project tackles the challenge of **restoring satellite images obscured by cloud cover**, significantly improving the usability of Earth observation data. By leveraging state-of-the-art deep learning techniques, the system enables better analysis for:

- 🌾 **Agricultural monitoring and land use classification**
- 🌳 **Vegetation health and environmental tracking**
- 🏙️ **Urban development and infrastructure planning**
- 🌊 **Natural disaster assessment and response**

### Key Technologies

The project combines two powerful deep learning approaches:

1. **ResNet-18 Classifier** — High-accuracy land-use classification
2. **Generative Adversarial Networks (GANs)** — Cloud artifact removal and image restoration using Pix2Pix and CycleGAN architectures

---

## ⚙️ Features

✅ **EuroSAT Dataset Integration** — High-quality RGB satellite imagery covering 10 distinct land-use classes  
✅ **ResNet-18 Classifier** — Achieves over **95% accuracy** on test data with transfer learning  
✅ **Pix2Pix & CycleGAN Architectures** — Advanced cloud removal and image-to-image translation  
✅ **Comprehensive Metrics** — SSIM & PSNR evaluation for restoration quality assessment  
✅ **Interactive Visualization** — Before/after comparison dashboard for result analysis  
✅ **Reproducible Environment** — Complete setup with virtual environment and dependency management  
✅ **Production-Ready Code** — Modular architecture with clean separation of concerns  

---

## 🧠 Model Architectures

### 🟢 ResNet-18 Classifier

A transfer learning approach for satellite image classification:

- **Base Model:** ResNet-18 pretrained on ImageNet
- **Fine-tuning:** Adapted for EuroSAT's 10 land-use classes
- **Optimizer:** Adam with learning rate of 1e-4
- **Loss Function:** CrossEntropyLoss
- **Performance:** 95.3% validation accuracy

### ☁️ GANs for Cloud Removal

Two complementary GAN architectures for image restoration:

#### **Pix2Pix**
- **Generator:** U-Net architecture with skip connections for detail preservation
- **Discriminator:** PatchGAN for realistic texture synthesis
- **Training:** Supervised paired image translation

#### **CycleGAN**
- **Architecture:** Dual generators and discriminators
- **Training:** Unsupervised domain translation with cycle consistency
- **Advantage:** No paired training data required

#### **Performance Metrics**
- **SSIM:** ~0.89 (structural similarity)
- **PSNR:** ~27.5 dB (signal quality)
- **Output Quality:** Visually realistic restoration with preserved texture and color fidelity

---

## 🧩 Project Structure

```
📦 satellite-image-enhancement/
├── 📂 src/
│   ├── 📄 dataset.py          # Dataset loader (EuroSAT & cloudy-clear pairs)
│   ├── 📄 model.py            # Model definitions (ResNet, GANs)
│   ├── 📄 train.py            # Training loops (classifier + GAN)
│   └── 📄 evaluate.py         # Evaluation & visualization tools
├── 📂 data/                   # Dataset directory (EuroSAT)
├── 📂 results/                # Generated results & sample outputs
├── 📄 main.py                 # Entry point for running experiments
├── 📄 notebook.ipynb          # Interactive analysis notebook
├── 📄 requirements.txt        # Project dependencies
├── 📄 model_eurosat.pth       # Saved model weights (optional)
├── 📄 LICENSE                 # Project license
└── 📄 README.md               # This file
```

---

## 🧰 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 4GB+ free disk space for dataset

### Quick Start

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yahia-frf/satellite-image-enhancement.git
cd satellite-image-enhancement

# 2️⃣ Create a virtual environment
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Download the EuroSAT dataset (optional - automatic on first run)
# The dataset will be downloaded automatically when you run training
```

---

## 🧪 Usage

### Training the ResNet Classifier

```bash
python main.py --mode train_classifier
```

**Options:**
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 1e-4)

### Training the GAN for Cloud Removal

```bash
python main.py --mode train_gan
```

**Options:**
- `--gan_type`: Choose between 'pix2pix' or 'cyclegan' (default: pix2pix)
- `--epochs`: Number of training epochs (default: 100)

### Evaluation and Visualization

```bash
python main.py --mode evaluate
```

This generates:
- Enhanced satellite images
- Quantitative metrics (SSIM, PSNR, accuracy)
- HTML visualization dashboard in `results/` folder

### Using the Jupyter Notebook

```bash
jupyter notebook notebook.ipynb
```

Interactive notebook for exploratory data analysis and visualization.

---

## 📈 Results Summary

| Model | Task | SSIM ↑ | PSNR (dB) ↑ | Accuracy ↑ |
|-------|------|--------|-------------|------------|
| **ResNet-18** | Classification | — | — | **95.3%** |
| **Pix2Pix** | Cloud Removal | **0.89** | **27.5** | — |
| **CycleGAN** | Cloud Removal | **0.87** | **26.8** | — |

### 🖼️ Visual Results

<div align="center">

**Before (Cloudy Input) → After (Enhanced Output)**

| Input Image | Output Image |
|:-----------:|:------------:|
| <img src="results/sample_cloudy.png" alt="Cloudy Input" width="300"/> | <img src="results/sample_enhanced.png" alt="Enhanced Output" width="300"/> |

*Sample restoration showing cloud removal while preserving landscape details*

</div>

---

## 🧮 Evaluation Metrics

- **SSIM (Structural Similarity Index):** Measures perceptual similarity between restored and ground truth images (range: 0-1, higher is better)
- **PSNR (Peak Signal-to-Noise Ratio):** Quantifies reconstruction quality in decibels (higher values indicate better quality)
- **Accuracy:** Classification performance on EuroSAT's 10 land-use categories

---

## 📚 Technologies & Dependencies

| Category | Tools / Libraries |
|----------|------------------|
| **Deep Learning** | PyTorch, Torchvision, CUDA |
| **Dataset** | EuroSAT (RGB version, 27,000 labeled images) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Evaluation** | scikit-image (SSIM, PSNR) |
| **Utilities** | NumPy, Pillow, tqdm, OpenCV |
| **Environment** | Python 3.8+, pip, virtualenv |

---

## 🗺️ Roadmap

- [ ] Add support for multispectral satellite imagery
- [ ] Implement attention mechanisms in GAN architectures
- [ ] Deploy as REST API for real-time inference
- [ ] Create web-based demo interface
- [ ] Expand to other satellite datasets (Sentinel-2, Landsat)
- [ ] Optimize for mobile/edge deployment

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{ferarsa2024satellite,
  author = {Ferarsa, Yahia},
  title = {Satellite Image Enhancement and Cloud Removal using GANs},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yahia-frf/satellite-image-enhancement}
}
```

---

## 🙏 Acknowledgments

- **EuroSAT Dataset:** Helber et al., "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
- **PyTorch Team:** For the excellent deep learning framework
- **Research Community:** For advancing GAN architectures and computer vision techniques

---

## 🧑‍💻 Author

<div align="center">

**Yahia Ferarsa**

🎓 Data Science & AI Student – École Nationale Polytechnique (ENP)  
🔬 Research Interests: Deep Learning, Computer Vision, Generative AI  

[![GitHub](https://img.shields.io/badge/GitHub-yahia--frf-181717?style=flat&logo=github)](https://github.com/yahia-frf)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/yahia-ferarsa)

</div>

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ and ☕ by Yahia Ferarsa

</div>
