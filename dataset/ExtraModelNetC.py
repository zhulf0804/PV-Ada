import numpy as np
import h5py
from torch.utils.data import Dataset



class ModelNetCExtraTest(Dataset):
    def __init__(self, h5_path):
        f = h5py.File(h5_path)
        self.data = f['data'][:].astype('float32')
        f.close()

    def __getitem__(self, item):
        pointcloud = self.data[item]
        return pointcloud

    def __len__(self):
        return self.data.shape[0]