import numpy as np
import open3d as o3d


def o3d_voxel_ds(cloud, voxel_size):
    return cloud.voxel_down_sample(voxel_size=voxel_size)


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    return pcd


def pcd2npy(pcd):
    npy = np.array(pcd.points).astype(np.float64)
    return npy
