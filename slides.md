# Slide 1 — Title
**Multi-Class Skin Lesion Classification using CNN**  
Fnu Sowrabh, Shrobanti Banerjee

---
# Slide 2 — Problem Statement
- Build a CNN to classify dermatoscopic images into 7 categories from HAM10000.
- Importance: early detection, assist dermatologists.

---
# Slide 3 — Dataset
- HAM10000, 10,015 images, 7 classes (akiec, bcc, bkl, df, mel, nv, vasc)
- Metadata includes age, sex, lesion id and location.

---
# Slide 4 — EDA: class distribution
- Show bar plot of counts per class (to be generated when dataset is available).

---
# Slide 5 — EDA: sample images
- Grid of sample images from each class (include examples).

---
# Slide 6 — Preprocessing
- Resize to 224x224, normalize, augment (flip, rotate, zoom, brightness).

---
# Slide 7 — Data split
- 70% train, 15% val, 15% test (stratified by class).

---
# Slide 8 — Model architecture
- Diagram: Conv(32)->BN->Pool -> Conv(64)->BN->Pool -> Conv(128)->BN->Pool -> Flatten -> Dense(256)->Dropout -> Dense(128)->Dropout -> Softmax(7)

---
# Slide 9 — Hyperparameters
- Learning rate: 1e-3 (tune: 5e-4, 1e-4)
- Batch size: 32 (tune: 16,64)
- Dropout: 0.5 (tune: 0.3,0.5)
- L2: 1e-4 (tune: 1e-3,1e-4)

---
# Slide 10 — Training details
- Optimizer: Adam, Loss: categorical_crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---
# Slide 11 — Results (simulated)
- Accuracy and loss curves: `demo_outputs/sim_accuracy.png`, `demo_outputs/sim_loss.png`

---
# Slide 12 — Confusion matrix (simulated)
- `demo_outputs/sim_confusion_matrix.png`

---
# Slide 13 — Discussion & Conclusions
- Limitations, next steps, ethical considerations, clinical deployment notes.

---
# Slide 14 — Code & Resources
- Repo: (you can upload this repo to GitHub). Contains train.py, model.py, data_loader.py, utils.py, demo_simulate.py

---
# Slide 15 — Thank you / Q&A
Contact: Shrobanti Banerjee
