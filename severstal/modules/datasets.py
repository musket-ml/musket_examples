from musket_core import image_datasets,datasets

@datasets.dataset_provider(origin="train.csv",kind="BinarySegmentationDataSet")
def get_segment1():
    return image_datasets.BinarySegmentationDataSet(["test_images","train_images"],"train.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train.csv",kind="BinarySegmentationDataSet")
def get_segment2():
    return image_datasets.BinarySegmentationDataSet(["test_images","train_images"],"train.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train.csv",kind="BinarySegmentationDataSet")
def get_segment3():
    return image_datasets.BinarySegmentationDataSet(["test_images","train_images"],"train.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train.csv",kind="BinarySegmentationDataSet")
def get_segment4():
    return image_datasets.BinarySegmentationDataSet(["test_images","train_images"],"train.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="classify.csv",kind="MultiClassificationDataset")
def get_classify():
    return image_datasets.MultiClassClassificationDataSet(["test_images","train_images"],"classify.csv","ImageId","ClassId")

