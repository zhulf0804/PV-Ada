import argparse
import h5py
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import ModelNetCExtraTest
from model import PCC


def test(args):
    device = torch.device('cuda')

    test_loader = DataLoader(
            ModelNetCExtraTest(h5_path=args.h5_path),
            batch_size=args.test_batch_size,
            shuffle=False,
            drop_last=False
        )
    
    model = PCC(voxel_size=args.voxel_size).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    test_pred = []
    with torch.no_grad():
        for pcd in tqdm(test_loader):
            pcd = pcd.to(device)
            pcd = pcd.permute(0, 2, 1)
            logits = model(pcd)
            preds = logits.argmax(dim=1)
            test_pred.append(preds.detach().cpu().numpy())

    test_pred = np.concatenate(test_pred)

    os.makedirs(args.saved_path, exist_ok=True)
    f = h5py.File(os.path.join(args.saved_path, 'results.h5'), 'w')
    f.create_dataset('label', data=test_pred)
    f.close()


if __name__ == '__main__':
    # Dataset settings
    parser = argparse.ArgumentParser(description='Extra ModelNet-C Prediction')
    parser.add_argument('--h5_path', type=str, default='/mnt/ssd1/lifa_rdata/PointCloud-C/cls_extra_test_data.h5',
                        metavar='N', help='Name of the experiment') 
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')              

    
    # Model settings
    parser.add_argument('--voxel_size', type=float, default=0.05, help='down sample voxel size')
    parser.add_argument('--ckpt', type=str, metavar='N', help='the trained checkpoint path') 

    # Saved settings
    parser.add_argument('--saved_path', type=str, help='the path to saved path')

    args = parser.parse_args()

    test(args)