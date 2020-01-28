import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score
from musket_core import metrics


@metrics.final_metric
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
    
    pred_raw0 = np.array(pred[:,0].tolist())
    pred_raw1 = np.array(pred[:,1].tolist())
    pred_raw2 = np.array(pred[:,2].tolist())
    
    argmax0 = np.argmax(pred_raw0, 1)
    argmax1 = np.argmax(pred_raw1, 1)
    argmax2 = np.argmax(pred_raw2, 1)
    
    pred0 = np.zeros(pred_raw0.shape, dtype = np.float32)
    pred1 = np.zeros(pred_raw1.shape, dtype = np.float32)
    pred2 = np.zeros(pred_raw2.shape, dtype = np.float32)
    
    for i in range(len(argmax0)):
        pred0[i,argmax0[i]] = 1
    
    for i in range(len(argmax1)):    
        pred1[i,argmax1] = 1
    
    for i in range(len(argmax2)):
        pred2[i,argmax2] = 1
    
    scores = []
    scores.append(recall_score(gt0, pred0, average='macro'))
    scores.append(recall_score(gt1, pred1, average='macro'))
    scores.append(recall_score(gt2, pred2, average='macro'))

    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score


@metrics.final_metric
def hmar_incorrect(fa, val):
    
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
