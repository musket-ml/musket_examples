'''
@author: 32kda
'''
from musket_core.fit_callbacks import after_fit
from musket_core import generic

@after_fit
def make_predictions():
    preds = generic.parse("questions1").predictions('test')
    preds.dump('predictions.csv')

if __name__ == '__main__':
    make_predictions()