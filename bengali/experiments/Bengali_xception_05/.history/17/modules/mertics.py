import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score


def hmar(fa, val):
    
    gt = []
    pred = []
    for item in val:
        gt.append(item.y)
        pred.append(item.prediction)
    
    gt = np.array(gt)
    pred = np.array(pred)
    
    gt0 = np.array(gt[:,0].tolist()).astype(np.float32)
    gt1 = np.array(gt[:,1].tolist()).astype(np.float32)
    gt2 = np.array(gt[:,2].tolist()).astype(np.float32)
    
    pred0 = (np.array(pred[:,0].tolist()) > 0.5).astype(np.float32)
    pred1 = (np.array(pred[:,1].tolist()) > 0.5).astype(np.float32)
    pred2 = (np.array(pred[:,2].tolist()) > 0.5).astype(np.float32)
    
    scores = []
    scores.append(recall_score(gt0, pred0, average='macro'))
    scores.append(recall_score(gt1, pred1, average='macro'))
    scores.append(recall_score(gt2, pred2, average='macro'))

    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score
