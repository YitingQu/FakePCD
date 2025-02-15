from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
import shutil
import torch
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
import random
from pathlib import Path

method2class = {'real': 0,
            'pointflow': 1,
            'diffusion': 2,
            'shapegf': 3,
            'pdgn': 4,
            'setvae': 4,
            'gnet': 4,
            'softflow':4}

def translate_pointcloud(pointcloud, translation_value):
    if not translation_value:
        xyz = np.random.uniform(low=-0.2, high=0.2, size=[3])
    else: 
        xyz = np.array([translation_value, translation_value, translation_value])
    translated_pointcloud = np.add(pointcloud, xyz).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud, theta=None):
    if not theta:
        theta = np.pi*2 * np.random.uniform() # rotating angle
    else:
        theta = np.deg2rad(theta)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

def load_pointclouds(data_dir, methods, shapes, partition):
    data_all, labels_all = [], []
    for method in methods:
        for shape in shapes:
            pc_files = glob.glob(os.path.join(data_dir, method, shape, partition, "*.npy"))
            for file in pc_files:
                pointcloud = np.load(file)
                label = method2class[method]
                if pointcloud.shape[0] == 4000:
                    continue
                if pointcloud.shape != (2048, 3):
                    print(method, pointcloud.shape)
                    pointcloud = pointcloud.T
                assert pointcloud.shape == (2048, 3)
                
                data_all.append(pointcloud)
                labels_all.append(label)
                    
    return data_all, labels_all

class ShapeNet2048(Dataset):
    def __init__(self, data_dir, methods, shapes, partition='train'):
        
        self.data, self.labels = load_pointclouds(data_dir, methods, shapes, partition)
        
    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        label = self.labels[idx]
        return pointcloud, label

    def __len__(self):
        return len(self.data)

class ShapeNet_Contrastive(Dataset):
    """
    Train: return a pc, augmented pc and the actual label
    Test: original pointcloud or augmented pc
    """
    def __init__(self, data_dir, methods, shapes, partition, args):
      
        self.translate = args.translate
        self.jitter = args.jitter
        self.rotate = args.rotate
        self.translation_value = args.translation_value
        self.sigma = args.sigma
        self.theta = args.theta
        
        self.data, self.labels = load_pointclouds(data_dir, methods, shapes, partition)

        self.partition = partition
        
    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        label = self.labels[idx]
        
        aug_type = np.random.randint(0,3)

        if self.partition == 'train':
            if aug_type == 0:
                aug_pointcloud = translate_pointcloud(pointcloud, 0.22)
            elif aug_type == 1:
                aug_pointcloud = jitter_pointcloud(pointcloud, 0.03)
            elif aug_type == 2:
                aug_pointcloud = rotate_pointcloud(pointcloud,theta=30)
            return pointcloud, aug_pointcloud, label
        
        elif self.partition == 'test':
            # if aug_type == 0:
            #     aug_pointcloud = translate_pointcloud(pointcloud, self.translation_value)
            # elif aug_type == 1:
            #     aug_pointcloud = jitter_pointcloud(pointcloud, self.sigma)
            # elif aug_type == 2:
            #     aug_pointcloud = rotate_pointcloud(pointcloud, self.theta)
            return pointcloud, label

    def __len__(self):
        return len(self.data)

class ShapeNet_Siamese(Dataset):
    """
    Train: for each sample creates randomly a positive or a negative pair
    Test: creates a fixed pairs for testing
    output：
    """
    def __init__(self, data_dir, methods, shapes, partition):
      
        self.data, self.labels = load_pointclouds(data_dir, methods, shapes, partition)
        
        self.data = list(zip(self.data, self.labels))
        
    def __getitem__(self, idx):
        pc0_tuple = self.data[idx]
        get_same_class = random.randint(0, 1) # 0 same class: 1: different class
        if get_same_class:
            while True:
                pc1_tuple = random.choice(self.data)
                if pc0_tuple[1] == pc1_tuple[1] and (pc0_tuple[0] != pc1_tuple[0]).any():
                    break
        else:
            while True:
                pc1_tuple = random.choice(self.data)
                if pc0_tuple[1] != pc1_tuple[1]:
                    break

        pc_0 = pc0_tuple[0]
        pc_1 = pc1_tuple[0]
        # 0: same class; 1: different class
        return pc_0, pc_1, torch.from_numpy(np.array([int(pc0_tuple[1]!=pc1_tuple[1])], dtype=np.int64))

    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":
    methods = ['real', 'pointflow', 'diffusion', 'shapegf', "pdgn", "setvae", "gnet", "softflow"]
    data = ShapeNet2048(data_dir="/home/c01yiqu/CISPA-projects/memes_multimodal-2022/FakeCloud/DGCNN/exp_data", methods=methods, shapes=["airplane"], partition="train")