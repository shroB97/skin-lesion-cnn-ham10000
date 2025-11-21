# Multi-Class Skin Lesion Classification using CNN (HAM10000)
Project repository scaffold created by ChatGPT for the SAT5165 final project.

**Contents**
- `data_loader.py` : dataset loading and preprocessing utilities (expects HAM10000).
- `model.py` : CNN architecture definition (Keras).
- `train.py` : training script with arguments for hyperparameters and augmentation.
- `utils.py` : helper functions for evaluation, plotting, class weights.
- `demo_simulate.py` : produces simulated training curves and sample confusion matrix image for demonstration when dataset/training is not available.
- `report.md` : draft of final project report (analysis, discussion, thoughts included).
- `slides.md` : 13+ slide markdown deck to convert into slides (pptx or reveal.js).
- `requirements.txt` : Python packages to install (suggested).

**Notes**
- This environment cannot download the HAM10000 dataset (internet disabled) and may not have heavy ML libraries installed.
- The `train.py` script is complete and ready: to run real training, download the dataset from Kaggle (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), place images under `data/` and run `python train.py --data_dir path/to/images`.
- A simulated demo is included to produce screenshots if training is not performed here.
