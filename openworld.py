import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from models import PointNet_Contrastive, PointNetCls, DGCNN_cls, DGCNN_Contrastive
import numpy as np
import random
from torch.utils.data import DataLoader
from utils import *
import sklearn.metrics as metrics
import sys
from pathlib import Path
from args import get_args
from dataset import ShapeNet2048, ShapeNet_Contrastive, ShapeNet_Siamese

torch.manual_seed(0) 
random.seed(0)
np.random.seed(0)


def load_model(args, device):
    #Try to load models
    num_sources = len(args.sources)
    output_channels = num_sources
    if args.model == 'pointnet_contrastive':
        if args.pretrained_path:
            model = PointNet_Contrastive(args)
            pretrained_dict = torch.load(args.pretrained_path)
            feat_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('feat.')}
            model.feat.load_state_dict(feat_dict)
            model.to(device)
        else:
            model = PointNet_Contrastive(args).to(device)
            
    elif args.model == 'dgcnn_contrastive':
        if args.pretrained_path:
            model = DGCNN_Contrastive(args)
            pretrained_dict = torch.load(args.pretrained_path)
            adjusted_state_dict = {k[len("feat."):]: v for k, v in pretrained_dict.items() if k.startswith("feat.")}
            model.feat.load_state_dict(adjusted_state_dict)
            model.to(device)
            
            # pretrained_model = DGCNN_cls(args, output_channels=output_channels)
            # pretrained_model = nn.DataParallel(pretrained_model)
            # pretrained_model.load_state_dict(torch.load(args.pretrained_path))
            # model = DGCNN_Contrastive(args, pretrained_model=pretrained_model).to(device)
        else:
            model = DGCNN_Contrastive(args).to(device)
    else:
        raise Exception("Not implemented")
    return model

def adjust_scheduler(opt, scheduler, args):
    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 1e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5

def train_contrastive(args, io, model, criterion, opt):
        
    train_loader = DataLoader(ShapeNet_Contrastive(data_dir=args.data_dir, methods=args.sources, shapes=args.shapes, partition='train', args=args),
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   drop_last=True)
    
    if args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    model.train()
    best_train_loss = float('inf')
    epochs_no_improve = 0
    patience = 100
    
    for epoch in range(args.epochs):
        train_loss = 0.0
        
        for idx, (pcs, aug_pcs, labels) in enumerate(train_loader):
            # pcs: (b_s, 2048, 3); label: (b_s, 1)
            labels = labels.unsqueeze(-1)
            pcs, aug_pcs = pcs.permute(0, 2, 1), aug_pcs.permute(0, 2, 1)
            inputs = torch.cat([pcs, aug_pcs], dim=0)
            labels = labels.repeat(2, 1).squeeze()
       
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device)

            opt.zero_grad()
            projections = model.forward_contrastive(inputs)
            loss = criterion(projections, labels)
            loss.backward()
            opt.step()

            train_loss += loss.item()
        outstr = 'Train %d, contrastive loss: %.6f' % (epoch, train_loss/(idx+1))
        io.cprint(outstr)

        adjust_scheduler(opt, scheduler, args)
        
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch}")
            torch.save(model.state_dict(), f"{args.save_dir}/{'_'.join(args.shapes)}_best.t7")
            break
        
        if (epoch !=0) and (epoch % args.save_freq == 0):
            torch.save(model.state_dict(), f"{args.save_dir}/{'_'.join(args.shapes)}_epoch{epoch}.t7")

def validation(io, test_loader, model, device):
    
    model = model.eval()

    test_acc = 0.0
    test_true = []
    test_pred = []
    res = {}
        
    for idx, (pc_0, pc_1, label) in enumerate(test_loader):
        pc_0, pc_1, label = pc_0.to(device), pc_1.to(device), label.to(device).squeeze()
        pc_0, pc_1 = pc_0.permute(0, 2, 1), pc_1.permute(0, 2, 1)
        logits = model(pc_0, pc_1)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_balance_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    confusion_mtx = metrics.confusion_matrix(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, test_balance_acc)
    io.cprint(outstr)

    res['acc'] = test_acc
    res['avg_acc'] = test_balance_acc
    res['confusion'] = confusion_mtx
    return res

def train(args, io):

    model = load_model(args, device=device)

    print(str(model))

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = SupervisedContrastiveLoss()
    criterion.to(device)
    # contrastive training
    train_contrastive(args, io, model, criterion, opt)

    # test_withContrastiveFeatures(args, device, 100, 85, 300)

if __name__ == "__main__":
    args = get_args()
    args.shapes=['airplane']
    args.sources=['real', 'pointflow', 'diffusion', 'shapegf']
    args.model='pointnet_contrastive'
    
    save_name = "_".join(args.shapes)

    args.save_dir = f'outputs/open_world/{args.model}/{save_name}'
    args.eval=False

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    io = IOStream(f"{args.save_dir}/logs.log")
    io.cprint(str(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(device)

    if not args.eval:
        train(args, io)
      