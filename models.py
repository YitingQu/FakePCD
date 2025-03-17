

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.batchnorm import BatchNorm1d

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2] # (b_s, 3, 2048), n_pts=2048
        trans = self.stn(x) # (b_s, 3, 2048) -> (b_s, 3, 3)
        x = x.transpose(2, 1) # (b_s, 3, 2048) -> (b_s, 2048, 3)
        x = torch.bmm(x, trans) # (b_s, 2048, 3)
        x = x.transpose(2, 1) # (b_s, 2048, 3) -> (b_s, 3, 2048)
        x = F.relu(self.bn1(self.conv1(x))) # (b_s, 3, 2048) -> (b_s, 64, 2048) 

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x # point feature after feature transform
        x = F.relu(self.bn2(self.conv2(x))) # (b_s, 64, 2048) -> (b_s, 128, 2048)
        x = self.bn3(self.conv3(x)) # (b_s, 128, 2048) -> (b_s, 1024, 2048) h(x)??
        hx = x.clone().detach()
        x = torch.max(x, 2, keepdim=True)[0] # (b_s, 1024, 2048) -> (b_s, 1024, 1) maxpool
        x = x.view(-1, 1024) # (b_s, 1024, 1) -> (b_s, 1024) global feature
        maxpool = x.clone().detach()
        if self.global_feat:
            # return x, trans, trans_feat # x:(b_s, 1024) global feature; hx: (b_s, 1024, 2048), maxpool: (b_s, 1024)
            return x, hx, maxpool
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, output_channels=2, feature_transform=False, visualize_critical=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.visualize_critical = visualize_critical
        
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def freeze_encoder(self):
        self.feat.requires_grad_(False)
    def forward(self, x):
        x, hx, maxpool = self.feat(x) # x:(b_s, 1024);  hx:(b_s, 1024, 2048) maxpool: global feature (b_s, 1024)
        x = F.relu(self.bn1(self.fc1(x))) # (b_s, 1024) -> (b_s, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # (b_s, 512) -> (b_s, 256)
        x = self.fc3(x)  # (b_s, 256) -> (b_s, output_channels)
        # return F.log_softmax(x, dim=1)
        if self.visualize_critical:
            return x, hx, maxpool
        else:
            return x # output x is logits of shape (b_s, output_channels)


class PointNet_Siamese(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNet_Siamese, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def freeze_encoder(self):
        self.feat.requires_grad_(False)

    def forward_once(self, x):
        x, _, _ = self.feat(x) # x:(b_s, 1024); trans: (b_s, 3, 3)
        x = F.relu(self.bn1(self.fc1(x))) # (b_s, 1024) -> (b_s, 512)
        x = F.relu(self.bn2(self.dropout(self.fc2(x)))) # (b_s, 512) -> (b_s, 256)
        return x
        
    def forward(self, pc0, pc1):
        out0 = self.forward_once(pc0)
        out1 = self.forward_once(pc1)
        dis = torch.abs(out0 - out1)
        out = self.sigmoid(self.fc3(dis)) # (b_s, 256) -> (b_s, 2)
        # out = self.fc3(dis)
        return out # 

class PointNet_Contrastive(nn.Module):
    def __init__(self, args):
        proj_dim = args.proj_dim
        super(PointNet_Contrastive, self).__init__()

        self.feat = PointNetfeat(global_feat=True)

        self.proj_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
            nn.BatchNorm1d(proj_dim))

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU())
        self.linear = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def freeze_encoder(self):
        self.feat.requires_grad_(False)
    
    def forward(self, x1, x2):
        x1 = self.forward_contrastive(x1)
        x2 = self.forward_contrastive(x2)
        # x = torch.cat([x1,x2])
        dist = torch.abs(x1-x2)
        return self.sigmoid(self.linear(dist))

    def forward_contrastive(self, x):
        x, _, _ = self.feat(x)
        x = self.proj_head(x) 
        x = F.normalize(x, dim=1)
        return x


class DGCNNfeat(nn.Module):
    def __init__(self, args):
        super(DGCNNfeat, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        hx = x.clone().detach()
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)
        maxpool = x.clone().detach()
        return x, hx, maxpool # maxpool (batch_size, emb_dims*2)

class DGCNN_Siamese(nn.Module):
    def __init__(self, args):
        super(DGCNN_Siamese, self).__init__()
        self.args = args
        self.k = args.k
        
        self.feat = DGCNNfeat(self.args)

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def freeze_encoder(self):
        self.feat.requires_grad_(False)
        
    def forward_once(self, x):
        x, hx, maxpool = self.feat(x)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        return x
        
    def forward(self, pc0, pc1):
        out0 = self.forward_once(pc0)
        out1 = self.forward_once(pc1)
        dist = torch.abs(out0 - out1)
        out = self.sigmoid(self.linear3(dist)) # (b_s, 256) -> (b_s, 2)
        return out 

class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=4, get_embeddings=False):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        self.get_embeddings = get_embeddings
        
        self.feat = DGCNNfeat(self.args)
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
    def freeze_encoder(self):
        self.feat.requires_grad_(False)
    def forward(self, x):
        x, hx, maxpool = self.feat(x)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        if self.get_embeddings:
            return x, hx, maxpool # x: (b_s, output_channels), hx:(b_s, emb_dim*2, 2048), maxpool: (b_s, emb_dim*2)
        else:
            return x # (b_s, output_channels)

class DGCNN_Contrastive(nn.Module):
    def __init__(self, args):
        super(DGCNN_Contrastive, self).__init__()
        self.args = args
        self.k = args.k
        
        self.feat = DGCNNfeat(self.args)

        self.proj_head = nn.Sequential(
            nn.Linear(args.emb_dims*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, args.proj_dim),
            nn.BatchNorm1d(args.proj_dim))

        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 5)

        self.linear = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        
    def freeze_encoder(self):
        self.feat.requires_grad_(False)

    def forward_contrastive(self, x):
        x, _, _ = self.feat(x)
        x = self.proj_head(x)
        x = F.normalize(x, dim=1)                       
        return x
        
    def forward(self, x1, x2):
        x1 = self.forward_contrastive(x1)
        x2 = self.forward_contrastive(x2)
        # x = torch.cat([x1,x2])
        dist = torch.abs(x1-x2)
        return self.sigmoid(self.linear(dist))