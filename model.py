"""
model.py
Defines the CNN architecture described in the project proposal.
"""
from tensorflow.keras import layers, models, regularizers

def make_cnn(input_shape=(224,224,3), num_classes=7, dropout_rate=0.5, l2_reg=0.0):
    model = models.Sequential()
    reg = regularizers.l2(l2_reg) if l2_reg>0 else None

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=reg, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=reg))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
