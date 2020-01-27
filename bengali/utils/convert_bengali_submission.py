import numpy as np
import pandas as pd
import tqdm
import imageio
import os


dir = "D:/kostya_work/runtime-New_configuration/TestProject/experiments/Bengali_0_01/predictions"
fileName = "BengaliTest00.[0]-pr.csv"

srcPath = os.path.join(dir,fileName)
fNamePlain = fileName[:fileName.index('.')]
dstPath = os.path.join(dir, fNamePlain + "_converted.csv")

srcDF = pd.read_csv(srcPath)
ids = srcDF['image_id']
l = len(ids)

items = []
for rec in srcDF.items():
    cTitle = rec[0]
    if cTitle == 'image_id':
        continue
    series = rec[1]
    for i in range(l):
        imId = ids[i]
        rowId = f"{imId}_{cTitle}"
        val = series[i]
        items.append({
            'row_id': rowId,
            'target': val
        })
    print("")
    pass

dstDF = pd.DataFrame(items, columns=['row_id', 'target'])
dstDF.to_csv(dstPath,index=False)