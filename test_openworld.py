import os
from re import A, X
from sklearn.utils import shuffle
from sklearn import svm, mixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from models import *
import numpy as np
import random
from torch.utils.data import DataLoader
from utils import *
import sklearn.metrics as metrics
from pathlib import Path
from args import get_args
from dataset import ShapeNet2048, ShapeNet_Contrastive, ShapeNet_Siamese

torch.manual_seed(0) 
random.seed(0)
np.random.seed(0)
_methods = ['real', 'pointflow', 'diffusion', 'shapegf']
# _methods = ['gnet', 'softflow']

method2class = {'real': 0,
            'pointflow': 1,
            'diffusion': 2,
            'shapegf': 3,
            'pdgn': 4,
            'setvae': 4,
            'gnet': 4,
            'softflow':4}

def load_model(args, model_path, device):
    #Try to load models
    if args.model == 'pointnet_contrastive':
        model = PointNet_Contrastive(args).to(device)
    elif args.model == 'dgcnn_contrastive':
        model = DGCNN_Contrastive(args).to(device)
    elif args.model == 'pointnet_siamese':
            model = PointNet_Siamese().to(device)
    elif args.model == 'dgcnn_siamese':
        model = DGCNN_Siamese(args).to(device)
    elif args.model == 'pointnet':
            model = PointNetCls(output_channels=len(_methods)).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args, output_channels=len(_methods)).to(device)
    else:
        raise Exception("Not implemented")
    model_dict = model.state_dict()
    checkpoint = torch.load(model_path)
    pretrained_dict = {k:v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict) # encoder + projection
    model.load_state_dict(model_dict)
    model = model.eval()
    model.freeze_encoder()
    return model

def get_features(args):
    test_methods = ['real', 'pointflow', 'diffusion', 'shapegf', 'pdgn', 'setvae']

    # load checkpoint 
    # Only load partial model beccause I modified the model structure
    print("=====> Resuming from checkpoint")
    model = load_model(args, args.model_path, device=device)

    # load test dataset
    test_loader = DataLoader(ShapeNet2048(data_dir=args.data_dir, methods=args.sources, shapes=args.shapes, partition='test'), 
                                          num_workers=4,
                                          batch_size=args.test_contrastive_batch_size, 
                                          shuffle=True, drop_last=True)
    all_embeddings = []
    all_labels = []
    for i, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device).squeeze()
        data = data.permute(0, 2, 1)
        embeddings, _, _ = model.feat(data)
        all_embeddings.extend(embeddings.detach().cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)

    np.savez(f"{args.base_dir}/embeddings_{'_'.join(args.shape)}_pdgn_setvae.npz", features=all_embeddings, labels=all_labels)

def test_withContrastiveFeatures(args, device, anchor_num=100, threashold_val=True):
    """
    calculate the (euclidean or cosine) distance between anchor samplesa and test samples
    """
    anchor_methods = ['real', 'pointflow', 'diffusion', 'shapegf']
    test_methods = ['real', 'pointflow', 'diffusion', 'shapegf', 'pdgn', 'setvae']
    new_label = len(anchor_methods)

    # Only load partial model beccause I modified the model structure
    print("=====> Resuming from checkpoint")
    model = load_model(args, args.contrastive_checkpoint, device)

    # get anchor data, concatenate all anchor methods
    anchors = get_anchors(args, anchor_methods, anchor_num) # anchors: dict of zip(pointcloud, labels)

    # get augmented anchors when robustness evaluation
    # anchors = get_augmented_anchors(args, anchor_methods, anchor_num)

    # select threshold
    threshold = select_threshold(args, anchors, anchor_num, model, device, args.percentile)
    print(args.percentile, threshold)

    
    labels_all = []
    predictions_all = []
    for test_method in test_methods:
        if threashold_val:
            _, _, test_features, test_labels = test_minibatch(args,test_method, model, device)
        else:
            test_features, test_labels, _, _ = test_minibatch(args,test_method, model, device)
        
        labels = method2class[test_method]
        labels_all.extend(test_labels)
        
        mean_dist_to_anchors = []
    
        for anchor_method in anchor_methods:
            anchor_features, _ = test_anchors_minibatch(args, anchors, anchor_method, model, \
                                                                        device, anchor_num)
            dist_matrix = calc_euclidean_distance(anchor_features, test_features)  # (anchor_num, test_sample_num)
            # dist_matrix = calc_cosine_distance(anchor_features, test_features)  # (anchor_num, test_sample_num)
            mean_dist = dist_matrix.mean(0) # the distance between one test method to one anchor method
            mean_dist_to_anchors.append(mean_dist)
        
        mean_dist_to_anchors = torch.stack(mean_dist_to_anchors) # (4, 96)
        min_values, min_indices = mean_dist_to_anchors.min(0)
        
        preds = min_indices
        for idx, value in enumerate(min_values):
            if value > threshold:
                preds[idx] = new_label
                
        print(preds)
            # # Determine labels based on the threshold
            # for idx, dist in enumerate(mean_dist):
            #     if dist < threshold:
            #         preds[idx] = method2class[anchor_method]  # Label as the current anchor method
        predictions_all.extend(preds)

    labels_all, predictions_all = np.array(labels_all), np.array(predictions_all)
        # labels_all, predictions_all = np.array(labels_all), np.array(predictions_all)
    close_world_indices = np.where(labels_all<len(anchor_methods))
    open_world_indices = np.where(labels_all>=len(anchor_methods))

    confusion_mtx = metrics.confusion_matrix(labels_all, predictions_all)
    close_accuracy = metrics.accuracy_score(labels_all[close_world_indices], predictions_all[close_world_indices])
    open_accuracy = metrics.accuracy_score(labels_all[open_world_indices], predictions_all[open_world_indices])
    close_f1 = metrics.f1_score(labels_all[close_world_indices], predictions_all[close_world_indices],average='weighted')
    open_f1 = metrics.f1_score(labels_all[open_world_indices], predictions_all[open_world_indices],average='weighted')

    print(f"close world accuracy: {close_accuracy}, open world accuracy: {open_accuracy}")
    print(f"close world f1 score: {close_f1}, open world f1 score: {open_f1}")
    print(confusion_mtx)
    
    # with open(f"{args.base_dir}/{'_'.join(args.shape)}_percentile{args.percentile}", 'wb') as fb:
    #     np.savez(fb, accuracy=(close_accuracy, open_accuracy), f1_score=(close_f1, open_f1), \
    #         distance=mean_dist_matrix, confusion=confusion_mtx)
    # return (close_accuracy, open_accuracy), (close_f1, open_f1)

def _test_one_method(args, anchors, test_method, test_samples_num, threshold, model): # num_anchor: number of batches in anchor samples
    """
    given a test_method, calculate the averaged similarities of it to its anchors (data, label)
    """
    test_loader = DataLoader(ShapeNet2048(test_method, 
                                        args.shape, partition='test', 
                                        num_points=args.num_points, 
                                        load_ratio=1, 
                                        load_mode=args.load_mode), 
                                        num_workers=4,
                                        batch_size=1, 
                                        shuffle=True, drop_last=True)
    test_acc = 0.0
    test_true = []
    test_pred = []
    similarity_matrix = [] # N test samples, 5 anchor methods
    new_label = len(anchors.keys())
    for idx, (test_pc, test_label) in enumerate(test_loader):
        # print(f'testing {test_method} now: {idx}')
        if idx >= test_samples_num:
            break
        # test_pc (1, 2048, 3)
        test_pc = test_pc.to(device)
        test_pc = test_pc.permute(0, 2, 1)
        test_batch = test_pc.repeat(args.test_contrastive_batch_size, 1, 1)
        one2N_sim = np.zeros(len(anchors.keys())) # similarity of one test pc to all anchor pcs (5,)
        for i, anchor_mtd in enumerate(anchors.keys()):
            anchors_pc, anchors_label = zip(*anchors[anchor_mtd])
            anchors_pc = torch.stack(anchors_pc)
            anchors_pc = anchors_pc.to(device).permute(0, 2, 1) # (anchor_num, 3, 2048)
            anchor_batch = torch.split(anchors_pc, args.test_contrastive_batch_size) # (split_num, split_batch_size, 3, 2048)
            # calculate the mean similarity between a single test pc and samples from one anchor_method
            raw_similarity = []
            for k in range(len(anchor_batch)):
                similarity = model.forward(test_batch, anchor_batch[k])[:, 0]
                raw_similarity.extend(similarity.detach().cpu().numpy())            
            one2N_sim[i] = np.mean(np.array(raw_similarity))            
            
        test_true.append(int(test_label.squeeze()))
        if np.max(one2N_sim) >= threshold:
            pred_label = np.where(one2N_sim==np.max(one2N_sim))[0][0] # a scalar
            test_pred.append(pred_label)
        else:
            test_pred.append(new_label) # represents a new class
            
        similarity_matrix.append(one2N_sim)

    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    similarity_matrix = np.array(similarity_matrix)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    
    print(f'test method: {test_method}   test accuracy: {test_acc}')

    return test_acc, similarity_matrix, test_true, test_pred

def test_withSiameseLogits(args, threshold=None):
    test_samples_num =100
    anchor_num = 100
    # threshold = 0.92 # manually select threshold to have a balanced accuracy
    anchor_methods = ['real', 'pointflow', 'diffusion', 'shapegf']
    test_methods = ['real', 'pointflow', 'diffusion', 'shapegf', 'pdgn', 'setvae']

    # Only load partial model beccause I modified the model structure
    print("=====> Resuming from checkpoint")
    model = load_model(args, args.model_path, device)

    # get anchors 
    anchors = get_anchors(args, anchor_methods, anchor_num)
    # select threshold
    if args.percentile:
        threshold = select_threshold_siamese(args, anchors, anchor_num, model, device, percentile=args.percentile)
    else:
        threshold = args.threshold
    print("threshold: ", threshold)

    accuracy = np.zeros(len(test_methods))
    similarity = []
    test_labels = []
    test_preds = []
    for i, test_mtd in enumerate(test_methods):
        accuracy[i], similarity_matrix, true, pred = _test_one_method(args, anchors, test_method=[test_mtd], \
                                                                        test_samples_num=test_samples_num, threshold=threshold, model=model)
        test_labels.extend(true)
        test_preds.extend(pred)
        similarity.append(np.mean(similarity_matrix, axis=0))
    confusion_mtx = metrics.confusion_matrix(test_labels, test_preds)
    similarity = np.array(similarity)
    test_labels, test_preds = np.array(test_labels), np.array(test_preds)

    close_world_indices = np.where(test_labels<len(anchor_methods))
    open_world_indices = np.where(test_labels>=len(anchor_methods))
    test_labels[open_world_indices] = len(anchors.keys())
    close_accuracy = metrics.accuracy_score(test_labels[close_world_indices], test_preds[close_world_indices])
    open_accuracy = metrics.accuracy_score(test_labels[open_world_indices], test_preds[open_world_indices])
    close_f1 = metrics.f1_score(test_labels[close_world_indices], test_preds[close_world_indices],average='weighted')
    open_f1 = metrics.f1_score(test_labels[open_world_indices], test_preds[open_world_indices],average='weighted')

    print(f"close world accuracy: {close_accuracy}, open world accuracy: {open_accuracy}")
    print(f"close world f1 score: {close_f1}, open world f1 score: {open_f1}")
    print(confusion_mtx)
    print(np.round(similarity, 3))

    with open(f"{args.base_dir}/pdgn_setvae_{'_'.join(args.shape)}_threshold{threshold}", 'wb') as fb:
        np.savez(fb, accuracy=(close_accuracy, open_accuracy), f1_score=(close_f1, open_f1), \
            similarity=similarity, confusion=confusion_mtx)


def differ_2openworlds_withSiameseLogits(args, threshold):
    test_methods = ['pdgn', 'setvae']
    # test_methods = ['real', 'pointflow', 'diffusion', 'shapegf', 'pdgn', 'setvae']

    # load checkpoint 
    print("=====> Resuming from checkpoint")
    model = load_model(args, args.contrastive_checkpoint, device)
    # load test dataset
    test_loader = DataLoader(ShapeNet_Siamese(test_methods, 
                                              args.shape, partition='test', 
                                              num_points=args.num_points, 
                                              load_ratio=1, 
                                              load_mode=args.load_mode), 
                                              num_workers=4,
                                              batch_size=args.test_contrastive_batch_size, 
                                              shuffle=True, drop_last=True)
    
    test_samples_num = 300
    test_true = []
    test_pred = []
    batch_num = int(test_samples_num/args.test_contrastive_batch_size)
    thresholds = torch.Tensor([threshold]).repeat(args.test_contrastive_batch_size).to(device)
    for idx, (test_pc0, test_pc1, test_label) in enumerate(test_loader):
        # print(f'testing {test_method} now: {idx}')
        if idx >= batch_num:
            break
        # test_pc (1, 2048, 3)
        test_pc0, test_pc1, test_label = test_pc0.to(device), test_pc1.to(device), test_label.to(device).squeeze()
        test_pc0, test_pc1 = test_pc0.permute(0, 2, 1), test_pc1.permute(0, 2, 1)
        logits = model.forward(test_pc0, test_pc1)
        similarity = logits[:, 0] # (b_s, 1)
        preds = torch.le(similarity, thresholds).type(torch.int32)
        test_true.extend(test_label.detach().cpu().numpy())
        test_pred.extend(preds.detach().cpu().numpy())
    
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    confusion_mtx = metrics.confusion_matrix(test_true, test_pred)
    print(test_acc)
    print(confusion_mtx)
    np.savez(f"{args.base_dir}/seperate_pdgn_setvae_{'_'.join(args.shape)}_withSiameseLogits.npz", \
                accuracy=test_acc, confusion=confusion_mtx)

def differ_2openworlds_withContrastiveFeatures(args):
    test_methods = ['pdgn', 'setvae']
    anchor_num = 100
    test_samples_num = 300

    # load model
    model = load_model(args, args.contrastive_checkpoint, device)  

    features = []
    labels = []
    for test_method in test_methods:
        test_features, test_labels = test_minibatch(args,test_method, model, device, test_samples_num)
        test_features = test_features.detach().numpy()
        features.extend(test_features)
        labels.append(test_labels.detach().numpy())
    features, labels = np.array(features), np.array(labels) # (len(test_methods)*test_samples_num, 128)
    # define cluster method here
    gm = mixture.GaussianMixture(n_components=int(len(test_methods)), random_state=1).fit(features)
    preds = gm.predict(features)
    preds = np.split(preds, len(test_methods))

    accuracy = {}
    from collections import Counter
    for cluster_idx, test_method in enumerate(test_methods):
        counter = Counter(preds[cluster_idx])
        most_common_num = counter.most_common(1)[0][1]
        accuracy[test_method] = (most_common_num/test_samples_num)
    print(accuracy)
    np.save(f"{args.base_dir}/differ2openworlds_{'_'.join(args.shape)}", np.array(accuracy))

def change_proj_dim(args):
    accuracy, f1_score = [], []
    for dim in [32, 64, 128, 256, 512]:
        args.proj_dim = dim
        args.contrastive_checkpoint=f"outputs/{args.model}/models/Dim_{dim}/car_setvae_epoch150.t7"
        acc, f1 = test_withContrastiveFeatures(args, device, 100, 300)
        accuracy.append(acc)
        f1_score.append(f1)
    print(accuracy, f1_score)
    np.savez(f"{args.base_dir}/change_proj_dim_{'_'.join(args.shape)}_percentile{args.percentile}_setvae.npz", \
                accuracy=np.array(accuracy), f1_score=np.array(f1_score))

def change_percentile(args):
    percentile = np.arange(70, 100, 2)
    accuracy, f1_score = [], []
    for perc in percentile:
        args.percentile = int(perc)
        acc, f1 = test_withContrastiveFeatures(args, device, 100, 300)
        accuracy.append(acc)
        f1_score.append(f1)
    print(accuracy, f1_score)
    np.savez(f"{args.base_dir}/change_percentile_{'_'.join(args.shape)}.npz", \
                accuracy=np.array(accuracy), f1_score=np.array(f1_score))

if __name__ == "__main__":
    # Training settings
    # os.environ['CUDA_VISIBLE_DEVICES']="5"
    args = get_args()
    args.shapes=['car']
    args.sources=['real', 'pointflow', 'diffusion', 'shapegf']
    args.model='pointnet_contrastive'
    # args.proj_dim=32
    # args.model_path='outputs/dgcnn_pdgn_setvae/models/airplane_epoch199.t7'
    args.contrastive_checkpoint='outputs/pointnet_contrastive/models/final_models/car_pdgn_setvae.t7'
    args.eval=True
    args.base_dir='outputs/pointnet_contrastive/tmp'
    args.percentile=80
    args.test_contrastive_batch_size=20

    device = "cuda" if torch.cuda.is_available() else "cpu"     
    torch.manual_seed(args.seed)
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    
    # change_proj_dim(args)
    # change_percentile(args)
    test_withContrastiveFeatures(args, device, anchor_num=100, threashold_val=True)
    # get_features(args)
    # differ_2openworlds_withSiameseLogits(args, 0.925)
    # differ_2openworlds_withContrastiveFeatures(args)
    # test_withSiameseLogits(args)
    
    