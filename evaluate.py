import argparse
import numpy as np
import pprint
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics

from dataset import ModelNet40, ModelNetC
from model import PCC


def eval(args):
    device = torch.device('cuda')
    
    test_loader = DataLoader(
        ModelNet40(
            data_root=args.modelnet_root,
            partition='test', 
            num_points=args.num_points), 
        num_workers=8,
        batch_size=args.test_batch_size, 
        shuffle=False, 
        drop_last=False)
    
    
    model = PCC(voxel_size=args.voxel_size).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    test_true, test_pred = [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    print('acc: ', test_acc)


def eval_corrupt_wrapper(model, fn_test_corrupt, args_test_corrupt):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    DGCNN_OA = {
        'clean': 0.926,
        'scale': 0.906,
        'jitter': 0.684,
        'rotate': 0.785,
        'dropout_global': 0.752,
        'dropout_local': 0.793,
        'add_global': 0.705,
        'add_local': 0.725
    }
    OA_clean = None
    perf_all = {'OA': [], 'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corrupt = {'OA': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, model=model, **args_test_corrupt)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}
            perf_corrupt['OA'].append(test_perf['acc'])
            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level
            pprint.pprint(test_perf, width=200)
            if corruption_type == 'clean':
                OA_clean = round(test_perf['acc'], 3)
                break
        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)
        if corruption_type != 'clean':
            perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - DGCNN_OA[corruption_type])
            perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (DGCNN_OA['clean'] - DGCNN_OA[corruption_type])
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        pprint.pprint(perf_corrupt, width=200)
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    perf_all['mOA'] = perf_all.pop('OA')
    pprint.pprint(perf_all, width=200)
    return perf_all


def eval_corrupt(args, model=None):
    device = torch.device('cuda')
    if model is None:
        model = PCC(voxel_size=args.voxel_size).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

    def test_corrupt(args, split, model):
        test_loader = DataLoader(
            ModelNetC(data_root=args.modelnetc_root, split=split),
            batch_size=args.test_batch_size, 
            shuffle=True, 
            drop_last=False
        )
        test_true = []
        test_pred = []
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits = model(data)
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}

    perf_all = eval_corrupt_wrapper(model, test_corrupt, {'args': args})
    return perf_all['mOA']


if __name__ == '__main__':
    # Dataset settings
    parser = argparse.ArgumentParser(description='Extra ModelNet-C Prediction')
    parser.add_argument('--modelnet_root', type=str, default='/mnt/ssd1/lifa_rdata/cls/modelnet40_ply_hdf5_2048')
    parser.add_argument('--modelnetc_root', type=str, default='/mnt/ssd1/lifa_rdata/PointCloud-C/modelnet_c')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')       
    parser.add_argument('--eval', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--eval_corrupt', action='store_true',
                        help='evaluate the model under corruption')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')       
    
    # Model settings
    parser.add_argument('--voxel_size', type=float, default=0.05, help='down sample voxel size')
    parser.add_argument('--ckpt', type=str, metavar='N', help='the trained checkpoint path') 

    args = parser.parse_args()

    if args.eval and args.eval_corrupt:
        raise ValueError('--eval and --eval_corrupt cannot be both specified')
    
    if args.eval:
        eval(args)
    elif args.eval_corrupt:
        eval_corrupt(args)