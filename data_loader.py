"""
data_loader.py
Utilities to prepare HAM10000 images for training.
Assumes images are stored in a directory with structure compatible with:
  data/<class_name>/*.jpg
or a csv metadata file that maps image ids to labels (the HAM10000 provides metadata csv).
"""
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_generators(data_dir, img_size=(224,224), batch_size=32, validation_split=0.15, seed=42):
    """
    Returns train, val, test generators using flow_from_directory.
    Requires directory structure: data_dir/train/<class>/img.jpg etc.
    Alternatively, modify this function to read the HAM10000 metadata csv and map images.
    """
    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        zoom_range=0.15,
        brightness_range=(0.8,1.2),
        validation_split=validation_split
    )
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )
    val_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )
    return train_gen, val_gen
