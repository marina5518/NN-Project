# 🗑️ Garbage Classification — Neural Network Project

> An image classification system that identifies garbage into 7 categories using two deep learning models, with a live interactive web app deployed on Streamlit.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://nn-project-wezk8czhyz9xhohvpyyfhf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Models](#models)
  - [Custom CNN (From Scratch)](#1-custom-cnn-from-scratch)
  - [MobileNetV2 (Transfer Learning)](#2-mobilenetv2-transfer-learning)
- [Results](#results)
- [Live Demo](#live-demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Contributors](#contributors)

---

## Overview

This project tackles the real-world problem of **automated garbage classification** using Convolutional Neural Networks. We built and compared two models:

1. A **custom CNN trained from scratch**
2. A **fine-tuned MobileNetV2** using transfer learning from ImageNet

Both models are deployed in a user-friendly Streamlit web app where users can upload an image and instantly receive a classification result.

---

## Dataset

The dataset contains images organized into **7 garbage categories**:

| Class | Description |
|-------|-------------|
| 🔋 Battery | Batteries and electronic cells |
| 📦 Cardboard | Cardboard boxes and packaging |
| 👕 Clothes | Textile and clothing waste |
| 🪟 Glass | Glass bottles, jars, and shards |
| 🔩 Metal | Metal cans, foil, and scraps |
| 📄 Paper | Newspapers, sheets, and paper products |
| 🧴 Plastic | Plastic bottles, bags, and containers |

The dataset was split as follows:

| Split | Ratio |
|-------|-------|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

---

## Data Preprocessing & Augmentation

Before training, the dataset underwent a preprocessing and augmentation pipeline to improve model generalization.

**Preprocessing steps:**
- Resizing all images to **224×224** pixels
- Pixel value normalization to the `[0, 1]` range (÷255)
- Organized into `train/`, `val/`, and `test/` directory splits with a fixed random seed (42) for reproducibility

**Augmentation strategy:**
To address class imbalance and improve robustness, augmentation was applied exclusively to the **training set**. Harder-to-classify classes (`Glass`, `Metal`, `Plastic`) received more aggressive augmentation.

| Augmentation | Base Classes | Hard Classes (Glass, Metal, Plastic) |
|---|---|---|
| Horizontal Flip | ✅ | ✅ |
| Rotation | ±10° | ±15° |
| Brightness Jitter | ±10% | ±15% |
| Contrast Jitter | ±10% | ±15% |
| Ops applied per image | 1–2 | 2–3 |
| Target multiplier | 1.2× | 1.3× |

---

## Models

### 1. Custom CNN (From Scratch)

A fully custom Convolutional Neural Network built using TensorFlow/Keras.

**Architecture:**

```
Input Image (224×224×3)
        ↓
Rescaling (1./255)
        ↓
Conv2D (32 filters, 3×3, ReLU, L2)
Conv2D (32 filters, 3×3, ReLU, L2)
BatchNormalization → MaxPooling2D
        ↓
Conv2D (64 filters, 3×3, ReLU, L2)
BatchNormalization → MaxPooling2D
        ↓
Conv2D (128 filters, 3×3, ReLU, L2)
BatchNormalization
Conv2D (128 filters, 3×3, ReLU, L2)
BatchNormalization → MaxPooling2D
        ↓
Dropout (0.4)
GlobalAveragePooling2D
Dense (128, ReLU, L2)
BatchNormalization → Dropout (0.5)
        ↓
Output: Dense (7, Softmax)
```

**Training configuration:**
- Optimizer: Adam (`lr = 3e-4`)
- Loss: Sparse Categorical Crossentropy
- Epochs: up to 40
- Callbacks: `EarlyStopping` (patience=5), `ReduceLROnPlateau` (factor=0.3, patience=3)

---

### 2. MobileNetV2 (Transfer Learning)

A two-phase transfer learning approach built on top of the **MobileNetV2** backbone pretrained on ImageNet.

**Architecture:**

```
Input Image (224×224×3)
        ↓
Normalization (÷255)
        ↓
MobileNetV2 Backbone (frozen — pretrained on ImageNet)
        ↓
GlobalAveragePooling2D
Dense (256, ReLU) → Dropout (0.5)
Dense (128, ReLU) → Dropout (0.3)
        ↓
Output: Dense (7, Softmax)
```

**Two-phase training:**

| Phase | Description | Learning Rate | Epochs |
|-------|-------------|---------------|--------|
| Phase 1 — Feature Extraction | All MobileNetV2 layers **frozen**, only the custom head is trained | `1e-3` | 30 |
| Phase 2 — Fine-Tuning | Last **30 layers** of MobileNetV2 unfrozen and retrained | `1e-5` | 20 |

**Training configuration:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Callbacks: `EarlyStopping` (patience=7), `ModelCheckpoint`, `ReduceLROnPlateau`

---

## Results

Both models were evaluated on the held-out test set using accuracy, loss, classification report (precision, recall, F1-score per class), and a confusion matrix heatmap.

| Model | Test Accuracy |
|-------|--------------|
| Custom CNN (from scratch) | See app |
| MobileNetV2 (fine-tuned) | See app |

> Full classification reports and confusion matrices are available in the Streamlit demo and the respective notebooks.

---

## Live Demo

🚀 **Try the app here:** [https://nn-project-wezk8czhyz9xhohvpyyfhf.streamlit.app/](https://nn-project-wezk8czhyz9xhohvpyyfhf.streamlit.app/)

Upload any image of garbage and the app will:
- Run it through both models
- Display the predicted class and confidence score
- Show a comparison between the CNN and MobileNetV2 predictions

---

## Project Structure

```
├── CNN_Model_Architecture.ipynb         # Custom CNN: architecture, training & evaluation
├── mobileNET_v2.ipynb                   # MobileNetV2: transfer learning & fine-tuning
├── Data_Preprocessing_Visualization.ipynb  # Dataset preparation, augmentation & EDA
├── NN_Data_set/
│   ├── train/
│   │   ├── Battery/
│   │   ├── Cardboard/
│   │   ├── Clothes/
│   │   ├── Glass/
│   │   ├── Metal/
│   │   ├── Paper/
│   │   └── Plastic/
│   ├── val/
│   └── test/
└── README.md
```

---

## Installation

To run locally, clone the repository and install dependencies:

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install tensorflow keras scikit-learn matplotlib seaborn pillow streamlit pandas

# Run the Streamlit app
streamlit run app.py
```

**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- Keras
- scikit-learn
- Matplotlib / Seaborn
- Pillow
- Streamlit
- Pandas / NumPy

---

## Contributors

This project was developed by:

| Name |
|------|
| 👩‍💻 Nesma Hesham |
| 👩‍💻 Menna Haitham |
| 👩‍💻 Nancy Alaa |
| 👩‍💻 Malak Ahmed |
| 👩‍💻 Nada Abdo |
| 👩‍💻 Marina Medhat |

---

<p align="center">Made with ❤️ using TensorFlow & Streamlit</p>
