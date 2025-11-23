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


---

## How to Run

1. Ensure dataset is downloaded and folder paths in `preprocess.py` are correct.
2. Run the training and evaluation script:

```bash
python train.py
'''
This will:

-Load and preprocess the data

-Compute class weights for imbalanced classes

-Build the CNN model (MobileNetV2 + custom top layers)

-Train the model with early stopping and learning rate scheduler

-Save the trained model (skin_cancer_tl_finetune.keras)

-Generate accuracy, loss plots, and confusion matrix

## Results

- **Validation Accuracy:** ~0.60

- **Classification Report:** precision, recall, f1-score per class

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| akiec | 0.46      | 0.51   | 0.49     |
| bcc   | 0.47      | 0.49   | 0.48     |
| bkl   | 0.41      | 0.47   | 0.43     |
| df    | 0.09      | 0.88   | 0.16     |
| mel   | 0.32      | 0.59   | 0.41     |
| nv    | 0.95      | 0.61   | 0.74     |
| vasc  | 0.43      | 0.86   | 0.58     |

- **Confusion Matrix:** `confusion_matrix_tl_finetune.png`  
- **Training Plots:** `accuracy_curve_tl_finetune.png`, `loss_curve_tl_finetune.png`

---

## Notes

- The dataset is highly imbalanced; **class weights** are applied to reduce bias.  
- **Data augmentation** (flips, rotations, brightness/contrast, zoom) is used to improve generalization.  
- **Transfer learning** with MobileNetV2 leverages pretrained features and reduces training time.  
- **Early stopping** and learning rate scheduling help prevent overfitting and optimize training.

