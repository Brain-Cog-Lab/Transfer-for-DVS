import os
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
import scipy.io as scio
import re
from tqdm import tqdm
import pickle
class CEPDVS(Dataset):
    """CEPDVS dataset
    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    length = 10000

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None, **kwargs):  # first 证明是否加载到了第一次
        super(CEPDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        classes = {
            'aquatic_mammals': 0,
            'fish': 1,
            'flowers': 2,
            'food_containers': 3,
            'fruit_and_vegetables': 4,
            'household_electrical_devices': 5,
            'household_furniture': 6,
            'insects': 7,
            'large_carnivores': 8,
            'large_man-made_outdoor_things': 9,
            'large_natural_outdoor_scenes': 10,
            'large_omnivores_and_herbivores': 11,
            'medium_mammals': 12,
            'non-insect_invertebrates': 13,
            'people': 14,
            'reptiles': 15,
            'small_mammals': 16,
            'trees': 17,
            'vehicles_1': 18,
            'vehicles_2': 19,
        }
        dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
        self.record_file = np.genfromtxt(open(os.path.join(save_to, 'pathFile.csv'), "rb"), delimiter=",",
                                         skip_header=1, dtype='U')
        file_path = os.path.join(save_to, "data/MAT/img/")
        first = kwargs['first'] if 'first' in kwargs else False
        if first:
            for path, dirs, files in os.walk(file_path):
                files.sort()
                for file in tqdm(files):
                    if file.endswith("mat"):
                        data = scio.loadmat(os.path.join(file_path, file), verify_compressed_data_integrity=False)
                        for key in data.keys():
                            if isinstance(data[key], np.ndarray):
                                data[key] = np.squeeze(data[key].astype(np.int64))
                                if key == "p":
                                    data[key] = np.where(data[key] == -1, 0, data[key])
                        self.data.append(np.fromiter(zip(data['x'], data['y'], data['ts'], data['p']), dtype=dtype))
                        label_name = re.search('(?:\S*\s){4}(\S+)', self.record_file[int(file[:-4])]).group(1)
                        self.targets.append(classes[label_name])
            with open(os.path.join(save_to, "data.pkl"), 'wb') as f:
                pickle.dump(self.data, f)
            with open(os.path.join(save_to, "targets.pkl"), 'wb') as f:
                pickle.dump(self.targets, f)
        else:
            with open(os.path.join(save_to, "data.pkl"), 'rb') as f:
                self.data = pickle.load(f)
            with open(os.path.join(save_to, "targets.pkl"), 'rb') as f:
                self.targets = pickle.load(f)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)