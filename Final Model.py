

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PATHS
# -----------------------------
BASE_PATH = "/Users/shrobantibanerjee/Downloads/archive"
IMG_DIR_1 = os.path.join(BASE_PATH, "HAM10000_images_part_1")
IMG_DIR_2 = os.path.join(BASE_PATH, "HAM10000_images_part_2")
METADATA = os.path.join(BASE_PATH, "HAM10000_metadata.csv")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

# -----------------------------
# LOAD METADATA
# -----------------------------
df = pd.read_csv(METADATA)
print("\nClass distribution:")
print(df['dx'].value_counts())

# Encode labels
le = LabelEncoder()
df["dx_idx"] = le.fit_transform(df["dx"])

# -----------------------------
# TRAIN/VAL/TEST SPLIT
# -----------------------------
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['dx_idx'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.176, stratify=train_df['dx_idx'], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["dx_idx"]),
    y=train_df["dx_idx"]
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("\nClass Weights:", class_weights)

# -----------------------------
# IMAGE PATH RESOLVER
# -----------------------------
def resolve_path(image_id):
    f1 = os.path.join(IMG_DIR_1, image_id + ".jpg")
    f2 = os.path.join(IMG_DIR_2, image_id + ".jpg")
    if os.path.exists(f1):
        return f1
    elif os.path.exists(f2):
        return f2
    else:
        raise FileNotFoundError(f"{image_id}.jpg not found!")

train_df["path"] = train_df["image_id"].apply(resolve_path)
val_df["path"] = val_df["image_id"].apply(resolve_path)
test_df["path"] = test_df["image_id"].apply(resolve_path)

# -----------------------------
# TF DATA PIPELINE
# -----------------------------
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

def make_dataset(df, training=False):
    paths = df["path"].values
    labels = df["dx_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2000)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# -----------------------------
# MODEL DEFINITION (TRANSFER LEARNING)
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(7, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("skin_cancer_tl_finetune.keras")
print("\nModel saved as skin_cancer_tl_finetune.keras")

# -----------------------------
# PLOTS
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig('accuracy_curve_tl_finetune.png')

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('loss_curve_tl_finetune.png')

print("Plots saved: accuracy_curve_tl_finetune.png, loss_curve_tl_finetune.png")

# -----------------------------
# TEST SET EVALUATION
# -----------------------------
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_tl_finetune.png")
print("Confusion matrix saved: confusion_matrix_tl_finetune.png")
