

from preprocess import load_metadata, split_data, add_paths, make_dataset, compute_class_weights
from model import create_cnn_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# -----------------------------
# LOAD DATA
# -----------------------------
df, le = load_metadata()
train_df, val_df, test_df = split_data(df)
train_df = add_paths(train_df)
val_df = add_paths(val_df)
test_df = add_paths(test_df)
class_weights = compute_class_weights(train_df)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print("Class Weights:", class_weights)

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# -----------------------------
# BUILD MODEL
# -----------------------------
model = create_cnn_model()
model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# -----------------------------
# TRAIN MODEL
# -----------------------------
EPOCHS = 20
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
print("Model saved: skin_cancer_tl_finetune.keras")

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
