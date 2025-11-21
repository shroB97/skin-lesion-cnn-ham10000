"""
train.py
Example training script. Usage:
  python train.py --data_dir path/to/data --epochs 30 --batch_size 32 --lr 1e-3
Notes:
- This script assumes data in directory structure compatible with ImageDataGenerator.flow_from_directory.
- For HAM10000 you may need to preprocess using the metadata csv to split into train/val/test folders.
"""
import argparse
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import make_cnn
from data_loader import get_image_generators
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--model_out", type=str, default="model.h5")
    return parser.parse_args()

def plot_history(history, out_path="training_curve.png"):
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title("Loss")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

if __name__=="__main__":
    args = parse_args()
    train_gen, val_gen = get_image_generators(args.data_dir, img_size=(args.img_size,args.img_size), batch_size=args.batch_size)
    model = make_cnn(input_shape=(args.img_size,args.img_size,3), num_classes=train_gen.num_classes, dropout_rate=args.dropout, l2_reg=args.l2)
    opt = Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(args.model_out, save_best_only=True, monitor='val_loss')
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)
    plot_history(history, out_path="training_curve.png")
    model.save(args.model_out)
