import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv('../input/train.csv')
    df.loc[:, 'kfold'] = -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    x = df.image_id.values
    y = df[['grapheme_root', 'vowel_diacritic', 'consonant diacritice']].values
    
    mskf = MultilabelStratifiedKFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(msk.split(x, y)):
        print('Train', trn_, 'Valid', val_)
        df.loc[val_, 'kfold'] = fold
    
    print(df.kfold.value_counts())
    df.to_csv('../input/train_folds.csv', index=False)