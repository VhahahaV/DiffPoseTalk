from .datasets import LmdbDataset, LmdbDatasetForSE
from .json_datasets import MotionJsonDataset, MotionInferenceDataset, MotionJsonDatasetForSE
from .json_datasets import MotionInferenceDataset, MotionJsonDataset


def infinite_data_loader(data_loader):
    while True:
        for data in data_loader:
            yield data
