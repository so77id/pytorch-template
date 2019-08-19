from __future__ import absolute_import, division, print_function

import torchvision

from datasets.video_dataset import VideoDataset

from utils.transformation import get_transformations
class DatasetFactory(object):
    """Dataset factory return instance of dataset specified in dataset_type."""
    @staticmethod
    def factory(dataset_type, *args, **kwargs):
        transform = get_transformations(dataset_type, *args, **kwargs)

        if dataset_type == "video":
            return VideoDataset(*args, **kwargs, transform=transform)
        if dataset_type == "blocks":
            return VideoDataset(*args, **kwargs, transform=transform)

        assert 0, "Bad dataset_type of dataset creation: " + dataset_type
