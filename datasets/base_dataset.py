import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def merge_labels(self, datasets):
        def fix_label_index(index_map, labels):
            new_labels = []
            for label in labels:
                new_labels.append(index_map[label])
            return np.array(new_labels)


        classes = []
        for dataset in datasets:
            classes.extend(dataset.classes)

        idxs = np.unique(np.array(classes), return_index=True)[1]
        classes = np.array([classes[idx] for idx in sorted(idxs)])
        for dataset in datasets:
            map = []
            new_labels = []
            for label in dataset.classes:
                map.append(np.where(classes == label)[0][0])
            map = np.array(map)

            dataset.label_map = map
            dataset.labels = fix_label_index(dataset.label_map, dataset.labels)
