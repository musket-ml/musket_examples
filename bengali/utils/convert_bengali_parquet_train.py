import numpy as np
import pandas as pd
import tqdm
import imageio
from musket_core import context

HEIGHT = 137
WIDTH = 236
SIZE = 256

p=context.get_current_project_data_path()


for i in range(4):
    df = pd.read_parquet(f"{p}/bengaliai-cv19/train_image_data_{i}.parquet")
    data0 = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    
    #data1 = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    for idx in tqdm.tqdm(range(len(df))):
        name = df.iloc[idx, 0]
        im0 = data0[idx]
        imageio.imwrite(f"{p}/bengaliai-cv19/train/{name}.jpg", im0)        
