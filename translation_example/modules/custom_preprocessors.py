from musket_core.datasets import PredictionItem
from musket_core.preprocessing import dataset_preprocessor, PreproccedPredictionItem
import numpy as np

Y_VOCAB_SIZE = 59985 #Hack, this should be made dynamic in future 

@dataset_preprocessor
def y_wrap(input_item:PredictionItem): #we need to expand dims for the output data since conv1d layer doesn't work without channels dimension
    #We should create PreproccedPredictionItem, if we want to have a link to the original item, like here - when we want to keep the original sentence    
    return PreproccedPredictionItem(input_item.id,input_item.x,np.expand_dims(input_item.y, len(input_item.y) - 1),input_item)