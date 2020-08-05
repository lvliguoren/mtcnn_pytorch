from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
import torch


mean_vals = [0.471, 0.448, 0.408]
std_vals = [0.234, 0.239, 0.242]


class WiderDataSet(Dataset):
    def __init__(self, data_dir='./data', pos_path=None, part_path=None, neg_path=None, landmark_path=None):
        super().__init__()
        self.data_dir = data_dir
        self.file_path = []

        if pos_path and os.path.exists(os.path.join(data_dir, pos_path)):
            with open(os.path.join(data_dir, pos_path)) as f:
                pos = f.readlines()
                self.file_path.extend(pos)

        if part_path and os.path.exists(os.path.join(data_dir, part_path)):
            with open(os.path.join(data_dir, part_path)) as f:
                part = f.readlines()
                self.file_path.extend(part)

        if neg_path and os.path.exists(os.path.join(data_dir, neg_path)):
            with open(os.path.join(data_dir, neg_path)) as f:
                neg = f.readlines()
                self.file_path.extend(neg)

        if landmark_path and os.path.exists(os.path.join(data_dir, landmark_path)):
            with open(os.path.join(data_dir, landmark_path)) as f:
                landmark = f.readlines()
                self.file_path.append(landmark[:])

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        file_info = self.file_path[idx].strip().split()
        file_name = file_info[0] # ../data/12/negative/0.jpg
        file_name = file_name.replace('../data/12', self.data_dir)
        img = cv2.imread(file_name)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=mean_vals, std=std_vals)(img)

        label = torch.tensor(int(file_info[1]))
        bbox_target = np.zeros((4, )).astype(np.float32)
        landmark_target = np.zeros((10,)).astype(np.float32)

        if len(file_info) == 6:
            bbox_target = np.array(file_info[2:6]).astype(np.float32)
        bbox_target = torch.from_numpy(bbox_target)
        if len(file_info) == 12:
            landmark_target = np.array(file_info[6:]).astype(np.float32)
        landmark_target = torch.from_numpy(landmark_target)

        return img, label, bbox_target, landmark_target