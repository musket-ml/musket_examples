'''
@author: 32kda
'''
from musket_core.fit_callbacks import after_fit
from musket_core import generic

@after_fit
def make_predictions():
    preds = generic.parse("pairs_bert").predictions('test')
    preds.dump('predictions_bert.csv')
    

if __name__ == '__main__':
    make_predictions()