import h5py
import numpy as np
import os
import glob
import math
import open3d
import torch
from torch.utils.data import Dataset

from utils import PointWOLF


def load_data(data_root, partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_root, f'ply_data_{partition}*.h5')):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


# not used
def twisting(x, sigma=1):
    '''
    Modified from pytorch implmentation in
    https://github.com/gaperezsa/3DeformRS/blob/develop/SmoothedClassifiers/Pointnet2andDGCNN/SmoothFlow.py#L418-L481

    twisting_matrix = 

    [[cos(alpha*z)    ,sin(alpha*z)      ,0],              [[cos(alpha*z)       ,cos(alpha*z + pi/2), 0]
    [-sin(alpha*z)   ,cos(alpha*z)      ,0]        =       [cos(alpha*z - pi/2),cos(alpha*z)       , 0] 
    [0               ,0                 ,1]]               [0                  ,0                  , 1]]
            
    '''
    n = x.shape[0]
    twisting_coeff = np.random.randn() * sigma
    
    z = np.repeat(x[:, 2], 4)
    alpha = np.repeat(twisting_coeff, 4*n)

    #create transformation matrixes
    bool_mask = np.tile([[1,1,0],[1,1,0],[0,0,0]], [n, 1, 1]).astype(np.bool)
    transformation_matrixs = np.tile(np.eye(3), [n,1,1])
    angles = alpha * z

    transformer = np.tile([0,math.pi/2,-math.pi/2,0], n)
    angles += transformer
    transformation_matrixs[bool_mask] = np.cos(angles)

    #use transformation to get twisted x
    twisted_x = np.matmul(transformation_matrixs, x[:, :, None])[:, :, 0]
    return twisted_x


# optional
def tapering(x, sigma=1):
    '''
    Modified from pytorch implmentation in
    https://github.com/gaperezsa/3DeformRS/blob/develop/SmoothedClassifiers/Pointnet2andDGCNN/SmoothFlow.py#L351-L416
    
    tapering_matrix = 
        [[0.5*a^2*z+b*z+1 ,0                 ,0],
        [0               ,0.5*a^2*z+b*z+1   ,0]
        [0               ,0                 ,1]].
    '''
    n = x.shape[0]
    tapering_coeff = np.random.randn(1, 2) * sigma
    
    z = np.repeat(x[:, 2], 2)
    a = np.repeat(tapering_coeff[:, 0], 2*n)
    b = np.repeat(tapering_coeff[:, 1], 2*n)

    bool_mask = np.tile([[1,0,0],[0,1,0],[0,0,0]], [n,1,1]).astype(np.bool)
    transformation_matrixs = np.tile(np.eye(3), [n,1,1])

    transformation_matrixs[bool_mask] = 0.5 * a**2  * z + b * z + 1
    tapering_x = np.matmul(transformation_matrixs, x[:, :, None])[:, :, 0]

    return tapering_x


class ModelNet40(Dataset):
    def __init__(self, data_root, num_points, partition='train', args=None):
        self.data, self.label = load_data(data_root, partition)
        self.num_points = num_points
        self.partition = partition
        self.PointWOLF = PointWOLF(args) if args is not None else None
        self.tapering = args.tapering if args is not None else None
        self.twisting = args.twisting if args is not None else None

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
            if self.PointWOLF is not None:
                _, pointcloud = self.PointWOLF(pointcloud)
            if self.twisting:
                pointcloud = twisting(pointcloud)
            if self.tapering:
                pointcloud = tapering(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]