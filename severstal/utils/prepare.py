'''
@author: 32kda
'''
import pandas as pd
import os

ds_path = '../data/severstal-steel-defect-detection'
data_frame = pd.read_csv(os.path.join(ds_path, 'train.csv'))
classes = 4
# Let's split our dataset into 4 datasets, one per each defect class
frames = [data_frame[data_frame['ClassId'] == (i + 1)] for i in range(classes)]
# We already splitted dataset into 4 parts, one per each class, so we don't need classId any longer
filtered_frames = [frame.drop('ClassId', axis=1) for frame in frames]
# Save classes into 4 independent csv files
for frm, idx in zip(filtered_frames, range(classes)): 
    frm.to_csv(os.path.join(ds_path, 'train' + str(idx + 1) + '.csv'), index=False)

classify_frame = data_frame.drop('EncodedPixels', axis=1)
existing = set(classify_frame['ImageId'].tolist())
img_path = os.path.join(ds_path, 'train_images')

image_list = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
for img in image_list:
    if img not in existing:
        classify_frame =  classify_frame.append(pd.Series({'ImageId':img,'ClassId' : '' }), ignore_index = True)
        
classify_frame.sort_values(by=['ImageId'])
classify_frame.to_csv(os.path.join(ds_path, 'classify.csv'), index = False)