import sys
import glob
import pandas as pd
import numpy as np
import albumentations
import torch
import joblib
from tqdm import  tqdm
from PIL import Image
import torch.nn funtional as F
import torch.nn as nn
import pretrainedmodels

TEST_BATCH_SIZE=32
MODEL_MEAN=(0.485, 0.456, 0.406)
MODEL_STD=(0.229, 0.224, 0.225)
IMG_HEIGHT=137
IMG_WIDTH=236
DEVICE='cuda'

class Resnet34(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)
    
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.feattures(x)
        x = F.adaptive_avg_pooling2d(x, 1).reshpae(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

class BengaliDatasetTest():
    def __init__(self, img_height, img_width, mean, std):
        df = pd.read_csv('../input/test.csv')
        self.img_ids = df.image_id.values
        self.img_arr = df.iloc[:, 1:].values

        self.aug = albumentations.Compose([
            albumentations.Resize(img_height, img_width, always_apply=True), 
            albumentations.Normalize(mean, std, always_apply=True)
        ])
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        image = self.img_arr[item]
        image = image.reshape(137, 236).astype(float)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image)['image']
        image = np.transpose(image, (2, 0, 1))
        return {
            'image':torch.tensor(image, dtype=torch.float), 
            'image_id':torch.tensor(self.img_ids[item], dtype=torch.long)
        }

model = Resnet34(pretrained=False)
model.load_state_dict(torch.load('../input/resnet34_fold0.bin'))
model.eval()
fold_results = dict()
for file_idx in range(4):
    df = pd.read_csv(f'../input/test_image_data_{file_idx}.parquet')
    dataset = BengaliDatasetTest(df=df, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, 
            mean=MODEL_MEAN, std=MODEL_STD)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, 
                                shuffle=False, num_workers=4)
    predictions = []
    for bi, d in enumerate(data_loader):
        image = d['image']
        img_id = d['image_id']
        image = image.to(DEVICE, dtype=torch.float)
        g, v, c = model(image)
        g = np.argmax(g, dim=1)
        v = np.argmax(v, axis=1)
        c = np.argmax(c, axis=1)
        for li, imid in enumerate(img_id):
            predictions.append((f'{imid}_grapheme_root', g[li]))
            predictions.append((f'{imid}_vowel_diacritic', v[li]))
            predictions.append((f'{imid}_consonant_diacritic', c[ii]))

    fold_results[file_idx] = predictions

final_predictions = np.zeros((len(predictions), 4))
for i, predictions in fold_results.items():
    for idx, input in predictions:
        final_predictions[idx, i] = input[1]


avg_preds = np.mean(final_predictions, axis=1)
prediction_submit = []
fold3_predictions = fold_results[3]
for i, prediction in enumerate(fold3_predictions):
    prediction_submit.append((prediction[0], avg_preds[i]))


sub = pd.DataFrame(prediction_submit, columns=['row_id', 'target'])




    
            
            
            
        


