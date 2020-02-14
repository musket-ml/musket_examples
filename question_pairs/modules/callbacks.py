'''
@author: 32kda
'''
from musket_core.fit_callbacks import after_fit
from musket_core import generic
from musket_core import projects
from pathlib import Path
import os


@after_fit
def make_predictions(): #We will make predictions for those experiments, which are already finished, but don't have the csv file with predictions
    experiments = projects.Project(Path(__file__).parent.parent).experiments()
    for exp in experiments:
        res_file = os.path.join(exp.path, exp.name() + '_result.csv')
        if exp.isCompleted() and not os.path.exists(res_file):
            preds = generic.parse(exp.path).predictions('test')
            preds.dump(res_file)

if __name__ == '__main__': #for debug only
    make_predictions()