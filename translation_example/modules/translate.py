'''
@author: 32kda
'''

from musket_core.fit_callbacks import after_fit
from musket_core import generic, utils, projects
from pathlib import Path
import os
import numpy as np
from musket_core import context

#Try translating small input file subset
@after_fit
def make_predictions():
    experiments = projects.Project(Path(__file__).parent.parent).experiments()
    for exp in experiments:
        if exp.isCompleted():
            file_path=os.path.join(context.get_current_project_data_path(),"rus.vocab")
            vocabulary=utils.load(file_path)            
            preds = generic.parse(exp.path).predictions('test')
            for item in preds:
                rootItem = item.rootItem()
                sentence = ''
                for indices in item.prediction:
                    sentence = sentence + " " + vocabulary.i2w[np.argmax(indices)]                    
                print(rootItem.x + " " +  sentence)

if __name__ == '__main__':
    make_predictions()