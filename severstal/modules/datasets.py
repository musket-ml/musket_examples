from musket_core import image_datasets,datasets

@datasets.dataset_provider(origin="train1.csv",kind="BinarySegmentationDataSet")
def get_segment1():
    return image_datasets.BinarySegmentationDataSet(["severstal-steel-defect-detection/test_images","severstal-steel-defect-detection/train_images"],"severstal-steel-defect-detection/train1.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train2.csv",kind="BinarySegmentationDataSet")
def get_segment2():
    return image_datasets.BinarySegmentationDataSet(["severstal-steel-defect-detection/test_images","severstal-steel-defect-detection/train_images"],"severstal-steel-defect-detection/train2.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train3.csv",kind="BinarySegmentationDataSet")
def get_segment3():
    return image_datasets.BinarySegmentationDataSet(["severstal-steel-defect-detection/test_images","severstal-steel-defect-detection/train_images"],"severstal-steel-defect-detection/train3.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="train4.csv",kind="BinarySegmentationDataSet")
def get_segment4():
    return image_datasets.BinarySegmentationDataSet(["severstal-steel-defect-detection/test_images","severstal-steel-defect-detection/train_images"],"severstal-steel-defect-detection/train4.csv","ImageId","EncodedPixels")

@datasets.dataset_provider(origin="classify.csv",kind="MultiClassificationDataset")
def get_classify():
    return image_datasets.MultiClassClassificationDataSet(["severstal-steel-defect-detection/test_images","severstal-steel-defect-detection/train_images"],"severstal-steel-defect-detection/classify.csv","ImageId","Class")

