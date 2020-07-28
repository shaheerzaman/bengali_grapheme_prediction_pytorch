import sys
sys.path.append('../src/')
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import BengaliDatasetTrain

dataset = BengaliDatasetTrain(folds=[0,1], img_height=137, img_width=236,
                            mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))

print(len(dataset))
idx = 0
img = dataset[idx]['image']
print(dataset[idx]['grapheme_root'])
print(dataset[idx]['vowel_diacritic'])
print(dataset[idx]['consonant_diacritic'])

img_arr = img.nump()
img_arr = np.transpose(img_arr, (0, 1, 2))
plt.imshow(img_arr)