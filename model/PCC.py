import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import local_operator, gather_points
from utils import npy2pcd, pcd2npy, o3d_voxel_ds


class VoxelLayer(nn.Module):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size

    def random_choice(self, x_item, l):
        n = x_item.shape[0]
        replace = n < l
        idx = np.random.choice(n, size=l, replace=replace)
        return x_item[idx]

    def voxel_core(self, x, voxel_size):
        bs, _, n = x.size()
        l, voxel_x = [], []
        for i in range(bs):
            pc1 = npy2pcd(x[i].transpose(1, 0).cpu().numpy())
            x_item = pcd2npy(o3d_voxel_ds(pc1, voxel_size))
            voxel_x.append(x_item)
            l.append(x_item.shape[0])
        l_mean = int(np.mean(l))
        voxel_x = np.array([self.random_choice(voxel_x[i], l_mean).transpose(1, 0) for i in range(bs)]) # (bs, 3, l_mean)
        voxel_x = torch.from_numpy(voxel_x).to(x)
        return voxel_x

    @torch.no_grad()    
    def forward(self, x):
        '''
        x: (b, 3, n)
        '''
        voxel_x1 = self.voxel_core(x, self.voxel_size)
        voxel_x2 = self.voxel_core(voxel_x1, 2 * self.voxel_size)
        return voxel_x1, voxel_x2
        


class PCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.1)

    
    def forward(self, x, k):
        '''
        x: (b, 3, n)
        return: (b, c, n)
        '''
        x = local_operator(x, k=k) # (b, 6, n, k)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.max(dim=-1, keepdim=False)[0] # (b, 64, n)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        x: (b, c, n)
        return: (b, c, n)
        '''
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PointTransformer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.conv_fuse = nn.Sequential(nn.Conv1d(5*channels, 4*channels, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(4*channels),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        '''
        x: (b, c, n)
        return: (b, 4c, n)
        '''
        batch_size, _, N = x.size()

        x0 = F.relu(self.bn1(self.conv1(x)))
        x0 = F.relu(self.bn2(self.conv2(x0)))
        x1 = self.sa1(x0)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x = torch.cat([x, x_cat], dim=1)
        x = self.conv_fuse(x)

        return x


class PCC(nn.Module):
    def __init__(self, voxel_size, output_channels=40):
        super().__init__()
        self.voxel_layer = VoxelLayer(voxel_size=0.05)
        self.pce = PCE()
        self.point_transformer = PointTransformer(channels=128)

        self.weight_layer1 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 1, 1)
        )
        self.weight_layer2 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 1, 1)
        )
        self.weight_layer3 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_fuse = nn.Sequential(
            nn.Linear(512*3, 512*2, bias=False),
            nn.BatchNorm1d(512*2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, output_channels)

    
    def forward(self, x, k=32):
        bs = x.size(0)
        voxel_x1, voxel_x2 = self.voxel_layer(x)
        
        x0 = self.point_transformer(self.pce(x, k=k)) # (b, 512, n1)
        x1 = self.point_transformer(self.pce(voxel_x1, k=k)) # (b, 512, n2)
        x2 = self.point_transformer(self.pce(voxel_x2, k=k)) # (b, 512, n3)

        n1, n2, n3 = x0.size(2), x1.size(2), x2.size(2)

        x0_w = self.sigmoid(self.weight_layer1(x0)) # (b, 1, n1)
        x1_w = self.sigmoid(self.weight_layer2(x1)) # (b, 1, n2)
        x2_w = self.sigmoid(self.weight_layer3(x2)) # (b, 1, n3)

        # x0 = gather_points(x0.permute(0, 2, 1).contiguous(), x0_w[:, :int(0.75 * n1)]).permute(0, 2, 1).contiguous()
        # x1 = gather_points(x1.permute(0, 2, 1).contiguous(), x1_w[:, :int(0.75 * n2)]).permute(0, 2, 1).contiguous()
        # x2 = gather_points(x2.permute(0, 2, 1).contiguous(), x2_w[:, :int(0.75 * n3)]).permute(0, 2, 1).contiguous()
        x0 = x0 * x0_w
        x1 = x1 * x1_w
        x2 = x2 * x2_w 

        x0 = F.adaptive_max_pool1d(x0, 1).view(bs, -1)
        x1 = F.adaptive_max_pool1d(x1, 1).view(bs, -1)
        x2 = F.adaptive_max_pool1d(x2, 1).view(bs, -1)

        x = torch.cat([x0, x1, x2], dim=1)
        x = self.conv_fuse(x)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
