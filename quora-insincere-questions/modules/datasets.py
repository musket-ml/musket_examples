from musket_text import text_datasets
from musket_core import datasets

@datasets.dataset_provider(origin="train.csv",kind="TextClassificationDataSet")
def get_sample():
    return text_datasets.BinaryTextClassificationDataSet("quora-insincere-questions-classification/train.csv","question_text","target")

@datasets.dataset_provider(origin="test.csv",kind="TextClassificationDataSet")
def get_test():
    return text_datasets.BinaryTextClassificationDataSet("quora-insincere-questions-classification/test.csv","question_text","target")

