from musket_core import datasets,genericcsv,context
from musket_core import image_datasets,datasets

@datasets.dataset_provider(origin="train.csv",kind="GenericDataSet")
def getBengali0():
    return image_datasets.MultiOutputClassClassificationDataSet("bengaliai-cv19/train", "bengaliai-cv19/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

# def getBengali1():
#     return image_datasets.MultiOutputClassClassificationDataSet("bengali/train_1", "bengali/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

@datasets.dataset_provider(origin="test.csv",kind="MultiClassificationDataset")
def getBengaliTest0():
    return image_datasets.MultiOutputClassClassificationDataSet("bengali/test_0", "bengali/test_flat.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

# @datasets.dataset_provider(origin="train.csv",kind="GenericDataSet")
# def getBengali0_small():
#     return image_datasets.MultiOutputClassClassificationDataSet("bengali/train_0", "bengali/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], len=10000)

import pandas as pd
import tqdm
import numpy as np

p=context.get_current_project_data_path()
HEIGHT = 137
WIDTH = 236


@datasets.dataset_provider(origin="test.csv",kind="MultiClassificationDataset")
def getData1():
    ds=pd.read_csv(f"{p}/bengaliai-cv19/train.csv")
    gr=ds["grapheme_root"].values
    vd=ds["vowel_diacritic"].values
    cd=ds["consonant_diacritic"].values
    for i in range(1):
        df = pd.read_parquet(f"{p}/bengaliai-cv19/train_image_data_{i}.parquet")
        data0 = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        
        class M(datasets.DataSet):
            
            
            def __len__(self):
                return len(df)
            
            
            def __getitem__(self, item)->datasets.PredictionItem:
                X=data0[item]
                
                y1=np.zeros(168)
                y1[gr[item]]=1
                
                y2=np.zeros(11)
                y2[vd[item]]=1
                
                y3=np.zeros(7)
                y3[cd[item]]=1
                return datasets.PredictionItem(item,np.stack([X],axis=-1),[y1,y2,y3])
            
        return M();

