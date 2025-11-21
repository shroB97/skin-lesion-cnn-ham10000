# Final Project Report — Multi-Class Skin Lesion Classification using CNN

**Authors:** Fnu Sowrabh, Shrobanti Banerjee  
**Course:** SAT5165 Final Project  
**Dataset:** HAM10000 (Kaggle)

---

## 1. Introduction & Problem Statement
We aim to build a CNN to classify dermatoscopic images into seven lesion types. The problem is both practically important (early detection of malignant lesions) and academically interesting due to imbalanced classes and small-sample issues per class.

## 2. Dataset
HAM10000 contains 10,015 dermatoscopic images and metadata fields (age, sex, localization, diagnosis). The images vary widely in illumination, resolution and device properties, which increases the domain gap.

## 3. EDA (Exploratory Data Analysis)
(When running with the dataset we will include:)
- Class distribution (bar plot)
- Age distribution (histogram)
- Example images per class (grid)
- Image size and brightness statistics

**Thoughts from EDA:** the dataset is significantly imbalanced — e.g., the 'nv' class dominates whereas classes such as 'df' and 'vasc' are rare. This requires careful handling (class weights, augmentation, possibly oversampling).

## 4. Preprocessing & Augmentation
Planned steps:
- Resize images to 224×224
- Normalize pixel values to [0,1]
- Augmentation: random flips, rotations, zoom, brightness jitter
- Use 70/15/15 train/val/test split (or stratified split using metadata CSV)
- Potentially use color normalization and hair removal / segmentation as advanced preprocessing (discussed in future work)

## 5. Model Design
We implement a custom CNN as described in the proposal:
- Conv blocks: 32 → 64 → 128 filters with BatchNorm and MaxPool
- Dense layers: 256, 128 with Dropout
- Output: softmax over 7 classes
- Loss: categorical crossentropy, Optimizer: Adam

**Why custom and not transfer learning?** Transfer learning (e.g., EfficientNet, ResNet) often gives stronger baselines. For the course project we implement a custom network to demonstrate understanding, but will also compare with transfer learning if resources permit.

## 6. Training Strategy & Hyperparameter Tuning
Hyperparameters to tune: learning rate, batch size, dropout, L2 weight decay, filter sizes, data augmentation strength.

We will use:
- EarlyStopping on validation loss
- ReduceLROnPlateau
- Class weights from training label frequencies

**Thoughts:** Given the class imbalance, we may test balanced sampling and focal loss as alternatives.

## 7. Results (Simulated demo)
Because training on the full dataset requires dataset download from Kaggle and substantial compute, we present simulated training curves and example confusion matrix to illustrate the expected outputs. When real training is run locally or on a cloud GPU, these will be replaced by actual results.

(see demo_outputs/ for simulated screenshots: `sim_accuracy.png`, `sim_loss.png`, `sim_confusion_matrix.png`, `sample_input.png`)

## 8. Discussion & Reflection
- Overfitting risk is the largest challenge; mitigation via augmentation and regularization is central.
- Model interpretability: Grad-CAM and saliency maps should be included to ensure clinically meaningful decision-making.
- Ethical considerations: Automated tools are decision-support only and must not replace clinical judgment. Dataset bias and demographic imbalance must be reported.

## 9. Next Steps
- Train the model on HAM10000 (requires dataset and GPU)
- Fine-tune a pretrained model for stronger baseline
- Add saliency visualization (Grad-CAM) and per-class ROC curves
- Share code via GitHub and prepare final slide deck

---
*This draft contains both the technical steps and our opinions on trade-offs. We'll update with real results when dataset and compute are available.*
