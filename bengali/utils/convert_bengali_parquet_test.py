import numpy as np
import pandas as pd
import tqdm
import imageio

HEIGHT = 137
WIDTH = 236
SIZE = 256

for i in range(4):
    df = pd.read_parquet(f"D:/datasets/bengaliai-cv19/test_image_data_{i}.parquet")
    data0 = df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    data1 = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
    for idx in tqdm.tqdm(range(len(df))):
        name = df.iloc[idx, 0]
        im0 = data0[idx]
        im1 = data1[idx]
        imageio.imwrite(f"D:/datasets/bengaliai-cv19/test_0/{name}.jpg", im0)
        imageio.imwrite(f"D:/datasets/bengaliai-cv19/test_1/{name}.jpg", im1)
