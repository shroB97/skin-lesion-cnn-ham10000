"""
demo_simulate.py
Generate simulated training curves and a sample confusion matrix image for inclusion in report/slides
Use this when you don't have the real dataset or computing resources handy.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

out_dir = "demo_outputs"
os.makedirs(out_dir, exist_ok=True)

# Simulate training history
epochs = np.arange(1, 21)
train_acc = 0.5 + (1 - np.exp(-0.15*epochs)) * 0.45 + np.random.normal(0,0.01,len(epochs))
val_acc = 0.45 + (1 - np.exp(-0.12*epochs)) * 0.40 + np.random.normal(0,0.02,len(epochs))
train_loss = 1.2*np.exp(-0.12*epochs) + np.random.normal(0,0.02,len(epochs))
val_loss = 1.4*np.exp(-0.10*epochs) + np.random.normal(0,0.03,len(epochs))

plt.figure()
plt.plot(epochs, train_acc, label='train_acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Simulated Accuracy Curves')
plt.savefig(os.path.join(out_dir,'sim_accuracy.png'))
plt.close()

plt.figure()
plt.plot(epochs, train_loss, label='train_loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Simulated Loss Curves')
plt.savefig(os.path.join(out_dir,'sim_loss.png'))
plt.close()

# Create simple confusion matrix image (fake numbers)
cm = np.array([[80,4,1,0,2,1,0],
               [3,70,2,2,1,0,0],
               [2,1,60,5,0,1,0],
               [0,0,5,55,2,1,0],
               [1,1,0,3,65,2,0],
               [0,0,0,1,2,90,0],
               [0,0,0,0,0,2,30]])

# draw a large image and write matrix values
img = Image.new('RGB', (800,600), color=(255,255,255))
d = ImageDraw.Draw(img)
title = "Simulated Confusion Matrix"
d.text((20,10), title, fill=(0,0,0))
cell_w = 80; cell_h = 40
classes = ['akiec','bcc','bkl','df','mel','nv','vasc']
start_x = 50; start_y = 60
# header
for j,c in enumerate(classes):
    d.text((start_x + cell_w*(j+1)+10, start_y-20), c, fill=(0,0,0))
for i,c in enumerate(classes):
    d.text((start_x-40, start_y + cell_h*(i)+10), c, fill=(0,0,0))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = str(cm[i,j])
        x = start_x + cell_w*(j+1) + 10
        y = start_y + cell_h*i + 10
        d.text((x,y), val, fill=(0,0,0))
img.save(os.path.join(out_dir,'sim_confusion_matrix.png'))

# Create a tiny sample image representing input
sample = Image.new('RGB', (224,224), color=(120,120,140))
draw = ImageDraw.Draw(sample)
draw.ellipse((50,40,174,164), fill=(200,80,80))
sample.save(os.path.join(out_dir,'sample_input.png'))

print("Demo artifacts saved to", out_dir)
