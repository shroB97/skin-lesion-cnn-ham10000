# Skin Lesion Classification using CNN (HAM10000)

## Overview
This project builds a **Convolutional Neural Network (CNN)** to classify dermatoscopic images into **7 skin lesion categories** using the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).  
The goal is to assist dermatologists in early detection of skin cancer using automated AI-based classification.

---

## Classes
The dataset includes the following 7 classes:

| Class Code | Diagnosis |
|------------|-----------|
| akiec      | Actinic keratoses and intraepithelial carcinoma |
| bcc        | Basal cell carcinoma |
| bkl        | Benign keratosis-like lesions |
| df         | Dermatofibroma |
| mel        | Melanoma |
| nv         | Melanocytic nevi |
| vasc       | Vascular lesions |

---

## Project Structure

skin-lesion-cnn/

├── main.ipynb
├── README.md
├── /datasets (auto-downloaded)
├── /HAM10000_images_part_1
├── /HAM10000_images_part_2
└── HAM10000_metadata.csv


---

---

## ** Code Overview**

This section summarizes all major steps performed in the notebook.

---
##  How to Run

- Install dependencies

- Run the full notebook:

'main.ipynb'

Dataset auto-downloads and the entire pipeline executes end-to-end.


## **1. Dataset Download & Setup**

- Dataset downloaded using `kagglehub`.
- Images copied into the working directory.
- `HAM10000_metadata.csv` loaded.
- Image paths mapped from two folders:
  - `HAM10000_images_part_1/`
  - `HAM10000_images_part_2/`

---

## **2. Data Exploration**

- Distribution of the 7 skin lesion classes is displayed.
- Random sample images per class visualized.

---

## **3. Train–Validation–Test Split**

- **Stratified split** to preserve class balance:
  - **70% Train**
  - **15% Validation**
  - **15% Test**

- Metadata updated with full file paths of each image.

---

## **4. Image Preprocessing & Data Generators**

Using `ImageDataGenerator`:

### **Training Generator**
- ResNet50 preprocessing
- Rotation, width/height shift
- Zoom, horizontal/vertical flip

### **Validation & Test Generators**
- Only preprocessing (no augmentation)

---

## **5. Class Imbalance Handling**

- Severe imbalance in classes like DF and VASC.
- Computed weights using:

```python
compute_class_weight('balanced', classes, y_train)
``` 

## **6. Model Architecture**

Base model: **ResNet50 (ImageNet-pretrained)**  
Two-phase training pipeline:

### **Stage 1: Frozen Base**
- Freeze all ResNet50 layers  
- Add custom classification head:
  - `GlobalAveragePooling2D`
  - `BatchNormalization`
  - `Dense(256, activation='relu')`
  - `Dropout(0.4)`
  - `Dense(7, activation='softmax')`

### **Stage 2: Fine-Tuning**
- Unfreeze layers from **layer 100 onward**
- Use a lower learning rate for stable training

---

## **7. Training**

Training occurs in two stages:

### **Stage 1**
- Learning rate: **1e-3**
- Only the custom head is trained (base frozen)

### **Stage 2**
- Learning rate: **1e-5**
- Fine-tune the ResNet50 base (from layer 100+)

### **Callbacks Used**
- `EarlyStopping`
- `ReduceLROnPlateau`

---

## **8. Evaluation**

Evaluation metrics are computed on **validation** and **test** sets:

- **Accuracy**
- **ROC-AUC (macro)**
- **Confusion Matrix**
- **Training History Plots** (Loss & Accuracy curves)

---

## **9. Performance Summary**

| **Metric**         | **Result** |
|--------------------|------------|
| Test Accuracy      | ~0.68      |
| Validation AUC     | ~0.94      |
| Test AUC           | ~0.94      |


---

## Author

**Shrobanti Banerjee**  
**Fnu Sowrabh**


