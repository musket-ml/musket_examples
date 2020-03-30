from musket_core import datasets, context
import os
from builtins import str

# Custom dataset for the eng-rus sentence pairs
class SrcToDest(datasets.DataSet):
    
    def __init__(self, path: str):
        inp_file = os.path.join(context.get_current_project_data_path(),path)
        data = to_pairs(load_doc(inp_file))
        self.src = [x[0] for x in data]
        self.dest = [x[1] for x in data]
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, item):
        return datasets.PredictionItem(item, self.src[item], self.dest[item])

@datasets.dataset_provider(origin="rus-eng/rus.txt",kind="GenericDataSet")
def get_train():
    return SrcToDest('rus-eng/rus.txt')

@datasets.dataset_provider(name='test',origin="rus-eng/rus_subset.txt",kind="GenericDataSet")
def get_test():
    return SrcToDest('rus-eng/rus_subset.txt')

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8', errors='ignore')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs