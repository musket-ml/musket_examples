from musket_core import datasets,genericcsv, preprocessing, context
import numpy as np

# We create separate dataset for the siamic network - with 2 outputs
class Questions2Outputs(datasets.DataSet):
    
    def __init__(self):
        self.data = context.csv_from_data('train.csv')
        self.q1 = self.data['question1'].values
        self.q2 = self.data['question2'].values
        self.target = self.data['is_duplicate'].values
        
    def __len__(self):
        return len(self.data)
    
    def get_questions(self, item):
        return [str(self.q1[item]),str(self.q2[item])]
    
    def __getitem__(self, item):
        return datasets.PredictionItem(item, self.get_questions(item), np.array([self.target[item]]))

@datasets.dataset_provider(origin="train.csv",kind="")
def get_train_siamic():
    return Questions2Outputs()

@datasets.dataset_provider(origin="train.csv",kind="GenericDataSet")
def get_train():
    return genericcsv.GenericCSVDataSet("train.csv",["question1","question2"],["is_duplicate"],[],{"question1":"as_is","question2":"as_is","is_duplicate":"binary"})

@datasets.dataset_provider(origin="test.csv",kind="GenericDataSet")
def get_test():
    return genericcsv.GenericCSVDataSet("test.csv",["question1","question2"],[])


@preprocessing.dataset_preprocessor
def preprocess(inp):
    return str(inp[0]) + " hello bear " + str(inp[1])