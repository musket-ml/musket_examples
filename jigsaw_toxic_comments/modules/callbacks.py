'''
@author: 32kda
'''
from musket_core.fit_callbacks import after_fit
from musket_core import generic

@after_fit
def make_predictions():
    preds = generic.parse("simpleCnn").predictions('Test')
    preds.dump('predictions_cnn.csv')
    preds = generic.parse("simpleRnn").predictions('Test')
    preds.dump('predictions_rnn.csv')

if __name__ == '__main__':
    make_predictions()