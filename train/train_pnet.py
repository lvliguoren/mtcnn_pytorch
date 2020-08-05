from model.mtcnn import PNet, ClS_Loss, Reg_Loss
from data.dataset import WiderDataSet
from torch.utils.data import DataLoader
import torch
import os
import cv2
from torchvision import transforms
import numpy as np
from model.mtcnn import PNet_Detect
from data.utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pnet_train_dataset = WiderDataSet(data_dir='E:/TEST/MTCNN_DATA/12', pos_path='pos_12.txt', part_path='part_12.txt', neg_path='neg_12.txt')
pnet_train_dataloader = DataLoader(pnet_train_dataset, batch_size=384, shuffle=True)


def main():
    pnet = PNet().to(device)
    optimizer = torch.optim.Adam(pnet.parameters(), lr=1e-3)
    critetion_cls = ClS_Loss(device)
    critetion_reg = Reg_Loss(device)

    pnet.train()
    for batch_idx, (img, label, bbox_target, landmark_target) in enumerate(pnet_train_dataloader):
        img = img.to(device)
        label = label.to(device)
        bbox_target = bbox_target.to(device)

        pred_cls, pred_reg, pred_cls_pro = pnet(img)

        cls_loss, accuracy, precision, recall = critetion_cls(pred_cls, label, pred_cls_pro)
        reg_loss = critetion_reg(pred_reg, label, bbox_target)

        loss = cls_loss + reg_loss*0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('batch_idx:{}, cls_loss:{:.4f}, reg_loss{:.4f}, loss:{:.4f}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}'
              .format(batch_idx, cls_loss, reg_loss, loss, accuracy, precision, recall))

        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.save(pnet, '../model/pnet.pth')

    print('train rnet complete')


@torch.no_grad()
def test():
    pnet = torch.load('../model/pnet.pth')
    pnet.eval()

    with open('E:/TEST/MTCNN_DATA/wider_face_train.txt') as f:
        lines = f.readlines()

    total_true = 0
    total_box = 0
    for line in lines:
        file_info = line.strip().split(' ')
        file_path = os.path.join('E:/TEST/MTCNN_DATA/WIDER_train/images', file_info[0]+ ".jpg")
        bbox_target = np.array(file_info[1:]).astype(np.float32).reshape(-1,4)

        pnet_detect = PNet_Detect(pnet, device)
        _, boxes_align = pnet_detect(file_path)

        pred_true = []
        for box in boxes_align:
            Iou = IoU(box, bbox_target)
            if np.max(Iou) > 0.5:
                pred_true.extend((Iou > 0.5).nonzero()[0])
        print('image {}, recall:{:.4f}%'.format(file_path,len(set(pred_true)) / bbox_target.shape[0] *100))
        total_true += len(set(pred_true))
        total_box += bbox_target.shape[0]
        print('avg recall:{:.4f}%'.format(total_true / total_box *100))



if __name__ == '__main__':
    # main()
    test()

