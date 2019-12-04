from musket_text import text_datasets
from musket_core import datasets

@datasets.dataset_provider(origin="train.csv",kind="TextClassificationDataSet")
def getTrain():
    return text_datasets.MultiClassTextClassificationDataSet("train.csv/train.csv","comment_text","toxic|severe_toxic|obscene|threat|insult|identity_hate")