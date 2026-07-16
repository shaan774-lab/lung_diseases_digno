# 🫁 Lung Diseases Diagnosis (Chest X-Ray Pneumonia Detection)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Deep Learning application that classifies chest X-ray images to detect **Pneumonia** using Transfer Learning with a fine-tuned **ResNet-18** model in PyTorch. The project features a clean, interactive web interface built with **Streamlit** for real-time inference.

---

## 🖥️ System Architecture & Interface Preview

Below is an infographic illustrating the general structure of the automated diagnosis workflow:

![Lung Disease Diagnosis System Dashboard](assets/lung_disease_dashboard.jpg)

---

## 📊 Technical Workflow

This flowchart describes the data processing, model training, and Streamlit inference pipeline:

```mermaid
graph TD
    %% Datasets & Preprocessing
    subgraph Data Pipeline
        A[Raw Chest X-Ray Images] --> B{Dataset Split}
        B -->|Train Set| C[Train Augmentation: Resize 224x224, Random Flip, Random Rotation]
        B -->|Test Set| D[Test Transform: Resize 224x224, Normalization]
    end

    %% Model and Training
    subgraph Deep Learning Engine
        C --> E[PyTorch DataLoader]
        E --> F[Pre-trained ResNet-18 Backbone]
        F --> G[Custom Fully Connected Classifier Layer]
        G --> H[Weighted Cross-Entropy Loss]
        H -->|Optimizes via Adam lr=1e-4| I[Trained weights: pneumonia_model_resnet18.pth]
    end

    %% Web App Inference
    subgraph Streamlit Web Application
        J[User Interface] -->|Uploads X-ray Image| K[PIL Image Preprocessing]
        I -->|Load Weights| L[ResNet-18 Inference Model]
        K -->|Forward Pass| L
        L --> M{Argmax Class Prediction}
        M -->|Class 0| N[NORMAL]
        M -->|Class 1| O[PNEUMONIA]
        N --> P[Display Results & Confidence Visualizer]
        O --> P
    end

    style Data Pipeline fill:#1a1c23,stroke:#3b82f6,stroke-width:2px,color:#ffffff
    style Deep Learning Engine fill:#1a1c23,stroke:#ee4c2c,stroke-width:2px,color:#ffffff
    style Streamlit Web Application fill:#1a1c23,stroke:#ff4b4b,stroke-width:2px,color:#ffffff
```

---

## ⚡ Key Features

* **Transfer Learning:** Fine-tuned ResNet-18 backbone pre-trained on ImageNet to leverage advanced feature extraction.
* **Weighted Class-Entropy:** Addressed dataset imbalance (Normal: 317 samples vs. Pneumonia: 855 samples) using computed class weights during training.
* **Interactive UI:** A lightweight web application built with Streamlit allowing doctors and users to drag-and-drop X-ray images.
* **Real-time Inference:** Rapid classifications (< 100ms) with clean normal/pneumonia status outputs.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.8+
* **Deep Learning Framework:** PyTorch & Torchvision
* **Frontend/Deployment:** Streamlit
* **Scientific Computing:** NumPy, OpenCV, Matplotlib, Scikit-learn
* **Utilities:** TQDM, Pillow

---

## 🚀 How to Run Locally

Follow these step-by-step instructions to set up the project on your machine:

### Step 1: Clone the Repository
```bash
git clone https://github.com/shaan774-lab/lung_diseases_digno.git
cd lung_diseases_digno
```

### Step 2: Create a Virtual Environment
We recommend using standard `venv` or `conda`:

**Using venv:**
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on macOS/Linux:
source venv/bin/activate
```

**Using Conda:**
```bash
conda create -p env python=3.8 -y
conda activate ./env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
Start the Streamlit server:
```bash
streamlit run app.py
```
This will automatically open the app in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
lung_diseases_digno/
│
├── assets/
│   └── lung_disease_dashboard.jpg       # Infographic diagram banner
│
├── notebook/
│   └── experiment.ipynb                 # Model exploration, data loading, and training script
│
├── app.py                               # Main Streamlit web application
├── requirements.txt                     # Python packages list
└── .gitignore                           # Git ignore configurations (ignores model weights and local data)
```

---

## 🧠 Model Training Details

The model was configured and trained in `notebook/experiment.ipynb` with the following parameters:
- **Optimizer:** Adam
- **Learning Rate:** `0.0001` (`1e-4`)
- **Loss Function:** `nn.CrossEntropyLoss` with weights adjusted to handle the dataset class balance.
- **Batch Size:** `32` (or configured per hardware capacity)
- **Epochs:** `10`
- **Output Weights:** Saved as `pneumonia_model_resnet18.pth`

> [!NOTE]
> The trained weights file `pneumonia_model_resnet18.pth` is excluded from the GitHub repository via `.gitignore` because of its file size. To run inference locally, you should place your trained model weights in the root directory.

---

## 👥 Contributors

* **Shaan Saxena** - [shaan774-lab](https://github.com/shaan774-lab)
