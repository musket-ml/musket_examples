'''
@author: 32kda
'''
import pandas as pd
import os
from musket_core.download_callbacks import after_download
from pathlib import Path

@after_download
def prepare_ds():    
    project_dir = Path(os.path.abspath(__file__)).parents[1]
    ds_path = os.path.join(project_dir,'data/severstal-steel-defect-detection')
    classify_path = os.path.join(ds_path, 'classify.csv')
    if os.path.exists(classify_path):
        print('classify.csv is already present - skipping')
        return
    print("Preparing classification dataset...")
    data_frame = pd.read_csv(os.path.join(ds_path, 'train.csv'))
        
    classify_frame = data_frame.drop('EncodedPixels', axis=1)
    existing = set(classify_frame['ImageId'].tolist())
    img_path = os.path.join(ds_path, 'train_images')
    
    image_list = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
    for img in image_list:
        if img not in existing:
            classify_frame =  classify_frame.append(pd.Series({'ImageId':img,'ClassId' : '' }), ignore_index = True)
            
    classify_frame.sort_values(by=['ImageId'])
    classify_frame.to_csv(classify_path, index = False)
    print("Done dataset preparation")
    pass
