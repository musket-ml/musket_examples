from musket_core import image_datasets,datasets

@datasets.dataset_provider(origin="train.csv",kind="BinarySegmentationDataSet")
def getTrain():
    return image_datasets.BinarySegmentationDataSet(["test/images","train/images"],"train.csv","id","rle_mask")

@datasets.dataset_provider(origin="sample_submission.csv",kind="BinarySegmentationDataSet")
def getTest():
    return image_datasets.BinarySegmentationDataSet(["test/images","train/images"],"sample_submission.csv","id","rle_mask")