from musket_core import datasets,genericcsv
from musket_core import image_datasets,datasets

@datasets.dataset_provider(origin="train.csv",kind="GenericDataSet")
def getBengali0():
    return image_datasets.MultiOutputClassClassificationDataSet("bengali/train_0", "bengali/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

def getBengali1():
    return image_datasets.MultiOutputClassClassificationDataSet("bengali/train_1", "bengali/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

@datasets.dataset_provider(origin="test.csv",kind="MultiClassificationDataset")
def getBengaliTest0():
    return image_datasets.MultiOutputClassClassificationDataSet("bengali/test_0", "bengali/test_flat.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'])

@datasets.dataset_provider(origin="train.csv",kind="GenericDataSet")
def getBengali0_small():
    return image_datasets.MultiOutputClassClassificationDataSet("bengali/train_0", "bengali/train.csv", 'image_id', ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], len=10000)
