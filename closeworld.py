import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from models import DGCNN_cls, PointNetCls
import numpy as np
import random
from torch.utils.data import DataLoader
from utils import cal_loss, IOStream
import sklearn.metrics as metrics
from args import get_args
from dataset import ShapeNet2048
from pathlib import Path
import tqdm

torch.manual_seed(0) 
random.seed(0)
np.random.seed(0)

def load_model(args, device):
    #load models
    num_sources = len(args.sources)
    output_channels = num_sources

    if args.model == 'pointnet':
        model = PointNetCls(output_channels=output_channels)
        model = model.to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args, output_channels=output_channels).to(device)
    else:
        raise Exception("Not implemented")
    return model

def train(args, io):
    
    train_loader = DataLoader(ShapeNet2048(data_dir=args.data_dir, methods=args.sources, shapes=args.shapes, partition='train'),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ShapeNet2048(data_dir=args.data_dir, methods=args.sources, shapes=args.shapes, partition='test'),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = load_model(args, device=device)
    # if args.model == 'dgcnn':
    #     model = nn.DataParallel(model)
    print(str(model))

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
    
    criterion = cal_loss
    best_train_loss = float('inf')
    epochs_no_improve = 0
    patience = 3
    for epoch in tqdm.tqdm(range(args.epochs)):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        mean_acc = 0
        model.train()

        num_bad = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze() # data (b_s, 2048, 3) label: (b_s)
            data = data.permute(0, 2, 1) # data: (b_s, 3, 2048)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data) # (b_s, output_channels)
            loss = criterion(logits, label)
            
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1] # (b_s)
            count += 1
            train_loss += loss.item() * batch_size
                
            batch_mean_acc = (preds==label).float().mean()

            mean_acc += batch_mean_acc

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        
 
        # test_true, test_pred = test(test_loader, model)
        # test_true = np.hstack(test_true)
        # test_pred = np.hstack(test_pred)
        # test_acc = metrics.accuracy_score(test_true, test_pred)
        
        # outstr = 'Epoch %d, loss: %.3f, train acc: %.3f, test acc: %.3f' % (epoch, train_loss/(count*args.batch_size), mean_acc/count, test_acc)
        outstr = 'Epoch %d, loss: %.3f, train acc: %.3f' % (epoch, train_loss/(count*args.batch_size), mean_acc/count)
        io.cprint(outstr)
        
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

def test(loader, model):
    model.eval()
    
    test_acc = 0.0
    test_true = []
    test_pred = []

    for data, label in loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.hstack(test_true)
    test_pred = np.hstack(test_pred)
    return test_true, test_pred
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    # f1_score = metrics.f1_score(test_true, test_pred, average='weighted')
    # confusion_mtx = metrics.confusion_matrix(test_true, test_pred)
    # print(test_acc, f1_score)
    # print(confusion_mtx)

if __name__ == "__main__":

    args = get_args()
    # args.shapes=['car']
    # args.sources=['real', 'pointflow', 'diffusion', 'shapegf']
    # args.model='pointnet'
    save_name = "_".join(args.shapes)

    args.save_dir = f'outputs/close_world/{args.model}/{save_name}'
    args.eval=False

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    io = IOStream(f"{args.save_dir}/logs.log")
    io.cprint(str(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(device)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
