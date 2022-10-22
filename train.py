import argparse
import numpy as np
import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics

from model import PCC
from dataset import ModelNet40
from loss import cal_loss
from utils import rsmix as rsmix_func
from evaluate import eval_corrupt


def main(args):
    if not os.path.exists(os.path.join(args.exp_name, 'models')):
        os.makedirs(os.path.join(args.exp_name, 'models'))

    train_loader = DataLoader(
        ModelNet40(
            data_root=args.modelnet_root, 
            partition='train', 
            num_points=args.num_points, 
            args=args if args.pw else None),
        num_workers=8, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True)
    test_loader = DataLoader(
        ModelNet40(
            data_root=args.modelnet_root,
            partition='test', 
            num_points=args.num_points), 
        num_workers=8,
        batch_size=args.test_batch_size, 
        shuffle=False, 
        drop_last=False)

    device = torch.device("cuda")

    model = PCC(
        voxel_size=args.voxel_size
        ).to(device)
    print(str(model))

    criterion = cal_loss

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    best_test_acc, best_modelnetc_acc = 0, 0
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        idx = 0
        total_time = 0.0
        tmp = []
        for data, label in tqdm(train_loader):
            '''
            implement augmentation
            '''
            rsmix = False
            r = np.random.rand(1)
            if (args.beta > 0 and r < args.rsmix_prob) or epoch > 300:
                rsmix = True
                data = data.cpu().numpy()
                data, lam, label, label_b = rsmix_func(data, label, beta=args.beta, n_sample=args.nsample,
                                                                 KNN=args.knn)
            if args.beta != 0.0:
                data = torch.FloatTensor(data)
            if rsmix:
                lam = torch.FloatTensor(lam)
                lam, label_b = lam.to(device), label_b.to(device).squeeze()
            data, label = data.to(device), label.to(device).squeeze()
            tmp.append(rsmix)
            if rsmix:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                start_time = time.time()
                logits = model(data)
                
                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - lam[i]) \
                               + criterion(logits[i].unsqueeze(0), label_b[i].unsqueeze(0).long()) * lam[i]
                    loss += loss_tmp
                loss = loss / batch_size

            else:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                start_time = time.time()
                logits = model(data)
                loss = criterion(logits, label)
            
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        print('rsmix: ', np.sum(tmp), '/', len(tmp))
        print('train total time is', total_time)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        
        print(outstr)

        if epoch >= 250:
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            total_time = 0.0
            model.eval()
            with torch.no_grad():
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device).squeeze()
                    data = data.permute(0, 2, 1)
                    batch_size = data.size()[0]
                    start_time = time.time()
                    logits = model(data)
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    loss = criterion(logits, label)
                    preds = logits.max(dim=1)[1]
                    count += batch_size
                    test_loss += loss.item() * batch_size
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
            model.train()
            print('test total time is', total_time)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                                test_loss * 1.0 / count,
                                                                                test_acc,
                                                                                avg_per_class_acc)

            print(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), os.path.join(args.exp_name, 'models', 'model.t7'))

        if epoch >= 300:
            model.eval()
            mOA = eval_corrupt(args, model=model)
            print('ModelNetC: ', mOA)
            if mOA > best_modelnetc_acc:
                best_modelnetc_acc = mOA
                torch.save(model.state_dict(), os.path.join(args.exp_name, 'models', 'modelnetc.t7'))
            torch.save(model.state_dict(), os.path.join(args.exp_name, 'models', 'model_final.t7'))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--modelnet_root', type=str, default='/mnt/ssd1/lifa_rdata/cls/modelnet40_ply_hdf5_2048',
                        metavar='N', help='Path to modelnet40')
    parser.add_argument('--modelnetc_root', type=str, default='/mnt/ssd1/lifa_rdata/PointCloud-C/modelnet_c',
                        metavar='N', help='Path to modelnet40C')
    parser.add_argument('--exp_name', type=str, default='PCC', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--eval_corrupt', type=bool, default=False,
                        help='evaluate the model under corruption')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')                   

    # added arguments
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=0.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512,
                        help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    # pointwolf
    parser.add_argument('--pw', action='store_true', help='use PointWOLF')
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point')
    parser.add_argument('--w_sample_type', type=str, default='fps',
                        help='Sampling method for anchor point, option : (fps, random)')
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25,
                        help='Maximum translation range of local transformation')

    # other augmentations
    parser.add_argument('--tapering', action='store_true', help='use tapering')
    parser.add_argument('--twisting', action='store_true', help='use twisting')
    
    # model
    parser.add_argument('--voxel_size', type=float, default=0.05, help='down sample voxel size')

    args = parser.parse_args()

    main(args)
