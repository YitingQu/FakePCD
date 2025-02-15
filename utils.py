import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import wasserstein_distance
from dataset import *

def calc_euclidean_distance(anchor_features, test_features):
    return torch.cdist(anchor_features, test_features) # (anchor_num, test_sample_num)

def calc_cosine_distance(anchor_features, test_features):
    anchor_norm = anchor_features / anchor_features.norm(dim=1)[:,None]
    test_norm = test_features / test_features.norm(dim=1)[:,None]
    cosine_distance = -torch.mm(anchor_norm, test_norm.transpose(0, 1))
    return cosine_distance 

def calc_wasserstein_distance(anchor_features, test_features):
    anchor_features, test_features = anchor_features.detach().numpy(), test_features.detach().numpy()
    distance = np.zeros((anchor_features.shape[0], test_features.shape[0]))
    for i, test_feat in enumerate(test_features):
        for j, anchor_feat in enumerate(anchor_features):
            dist = wasserstein_distance(test_feat, anchor_feat)
            distance[j][i] = dist
    return distance

def cal_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. 
    label smoothing: prevent overfitting and better generalize
    '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1) # (b_s, num_classes) # cuda error??
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) # 
        log_prb = F.log_softmax(pred, dim=1) # 

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean') # cuda error?

    return loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362
        :param temperature: int, default: 0.07
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def get_anchors(args, anchor_methods, anchor_num):
    data = {} # dict: key: method, values: (pc, label)
    for method in anchor_methods:
        anchor_loader = DataLoader(ShapeNet2048(data_dir=args.data_dir, methods=[method], shapes=args.shapes, partition='train'),
                                                batch_size=anchor_num, 
                                                shuffle=False, 
                                                drop_last=True)
        pcs, labels = next(iter(anchor_loader))
        data[method] = list(zip(pcs, labels))
    return data 

def test_minibatch(args, method, model, device, val_num=100):
    """
    test minibatch of test samples. Use minibatch because a larger samples_num comsumes too much memory
    """
    from torch.utils.data import DataLoader, random_split

    dataset = ShapeNet_Contrastive(data_dir=args.data_dir, methods=[method], shapes=args.shapes, partition='test', args=args)
    dataset_size = len(dataset)

    # Split the dataset into validation and test sets
    val_set, test_set = random_split(dataset, [val_num, dataset_size - val_num])
    
    # Create DataLoaders for each set
    val_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Function to process each set
    def process_set(loader):
        features = []
        labels = []
        for samples, _labels in loader:
            samples, _labels = samples.to(device=device, dtype=torch.float), _labels.to(device).squeeze()
            samples = samples.permute(0, 2, 1)
            _features = model.forward_contrastive(samples)  # (batch_size, 128)
            features.append(_features)
            labels.append(_labels)
        features = torch.cat(features, dim=0).cpu()  # (samples_num, 128)
        labels = torch.cat(labels, dim=0).cpu()
        return features, labels

    # Process validation and test sets
    val_features, val_labels = process_set(val_loader)
    test_features, test_labels = process_set(test_loader)

    return test_features, test_labels, val_features, val_labels

def test_anchors_minibatch(args, anchors, method, model, device, anchor_num):
    anchor_samples, anchor_labels = zip(*anchors[method])
    anchor_samples = torch.stack(anchor_samples)
    anchor_labels = torch.stack(anchor_labels)
    anchor_samples = anchor_samples.to(device).permute(0, 2, 1)
    anchor_batch_num = int(anchor_num // args.batch_size)
    anchor_samples = torch.split(anchor_samples, args.batch_size)
    anchor_features = []
    for k in range(anchor_batch_num):
        # _features, _, _ = model.feat(anchor_samples[k])
        _features = model.forward_contrastive(anchor_samples[k]) # (batch_size, 128)
        anchor_features.append(_features)
    anchor_features = torch.cat(anchor_features, dim=0).cpu() # (anchor_num, 128)
    return anchor_features, anchor_labels

def select_threshold(args, anchors, anchor_num, model, device, percentile):
    """
    select eulidean distance threshold
    """
    all_radius = []
    for j, anchor_method in enumerate(anchors.keys()):
        anchor_samples, anchor_labels = zip(*anchors[anchor_method])
        anchor_samples = torch.stack(anchor_samples)
        anchor_samples = anchor_samples.to(device).permute(0, 2, 1)
        anchor_batch_num = int(anchor_num // args.batch_size)
        anchor_samples = torch.split(anchor_samples, args.batch_size)
        anchor_features = []
        for k in range(anchor_batch_num):
            _features = model.forward_contrastive(anchor_samples[k]) # (batch_size, 128)
            anchor_features.append(_features)
        anchor_features = torch.cat(anchor_features, dim=0).cpu() # (anchor_num, 128)
        centroid = anchor_features.mean(dim=0).unsqueeze(0) # (1, 128)
        anchors_centroid_dist = torch.cdist(centroid, anchor_features).squeeze() # (anchor_num)
        radius = np.percentile(anchors_centroid_dist.detach().numpy(), percentile)
        all_radius.append(radius)
    return np.max(all_radius) # a selected distance

def select_threshold_siamese(args, anchors, anchor_num, model, device, percentile):
    """
    select similarity threshold when attribute with simple siamese learning
    """
    print("Finding thresholds ...")
    all_radius = []
    for j, anchor_method in enumerate(anchors.keys()):
        anchor_samples, anchor_labels = zip(*anchors[anchor_method])
        anchor_samples = torch.stack(anchor_samples)
        anchor_samples = anchor_samples.to(device).permute(0, 2, 1) # (anchor_num, 3, 2048)
        anchor_batch = torch.split(anchor_samples, args.batch_size)
        self_similarity = []
        for n in range(anchor_num):
            single_pc_batch = anchor_samples[n].repeat(args.batch_size, 1, 1)
            mean_similarity = []
            for k in range(len(anchor_batch)):
                similarity = model.forward(single_pc_batch, anchor_batch[k])[:, 0]
                mean_similarity.extend(similarity.detach().cpu().numpy())
            self_similarity.append(np.mean(np.array(mean_similarity)))

        self_similarity = np.array(self_similarity)
        radius = np.percentile(self_similarity, percentile)
        all_radius.append(radius)
    print("Threshold Found!")
    return np.min(all_radius) # a selected distance

