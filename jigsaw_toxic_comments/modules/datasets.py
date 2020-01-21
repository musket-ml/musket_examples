from musket_text import text_datasets
from musket_core import datasets

@datasets.dataset_provider(origin="train.csv",kind="TextClassificationDataSet")
def get_train():
    return text_datasets.MultiClassTextClassificationDataSet("train.csv","comment_text","toxic|severe_toxic|obscene|threat|insult|identity_hate")
@datasets.dataset_provider(origin="test.csv",kind="TextClassificationDataSet")
def get_test():
    return text_datasets.MultiClassTextClassificationDataSet("test.csv","comment_text","toxic|severe_toxic|obscene|threat|insult|identity_hate")