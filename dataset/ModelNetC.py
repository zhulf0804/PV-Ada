import os
import h5py
from torch.utils.data import Dataset


def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


class ModelNetC(Dataset):
    def __init__(self, data_root, split):
        h5_path = os.path.join(data_root, split + '.h5')
        self.data, self.label = load_h5(h5_path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
