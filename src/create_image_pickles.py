import pandas as pd
import joblib
import glob
from tqdm import tqdm

files = glob.glob('../input/train_*.parquet')
for file in files:
    df = pd.read_parquet(file)
    image_id = df.image_id.values
    df = df.drop('image_id', axis=1)
    image_array = df.values
    for j, img_id in tqdm(enumerate(image_id), total=len(image_id)):
        joblib.dump(image_array[j, :], f'../input/image_pickles/{img_id}.pkl')

