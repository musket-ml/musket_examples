'''
Created on 30 мар. 2020 г.

@author: 32kda
'''

from musket_core.fit_callbacks import after_fit
from musket_core import generic, utils, projects, builtin_datasets
from pathlib import Path
import os
import numpy as np
from musket_core import context

# Translate passed string
def translate(sentence:str):
    file_path=os.path.join(context.get_current_project_data_path(),"rus.vocab")
    vocabulary=utils.load(file_path)            
    preds = generic.parse('eng_to_ru').predictions(builtin_datasets.from_array([sentence], ['']))
    for item in preds:
        rootItem = item.rootItem()
        sentence = ''
        for indices in item.prediction:
            sentence = sentence + " " + vocabulary.i2w[np.argmax(indices)]                    
        print(rootItem.x + " " +  sentence)
        
if __name__ == '__main__':
    translate('Why do I need to be in Boston?')