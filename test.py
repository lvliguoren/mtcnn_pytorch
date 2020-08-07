from model.mtcnn import PNet, RNet, ONet, PNet_Detect, RNet_Detect, ONet_Detect
from data.dataset import WiderDataSet
from torch.utils.data import DataLoader
import torch
import os
import cv2
from torchvision import transforms
import numpy as np
from model.mtcnn import PNet_Detect
from data.utils import *


@torch.no_grad()
def main(img_path, pnet_path, rnet_path, onet_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pnet = torch.load(pnet_path).to(device)
    rnet = torch.load(rnet_path).to(device)
    onet = torch.load(onet_path).to(device)
    pnet.eval()
    rnet.eval()
    onet.eval()
    pnet_detect = PNet_Detect(pnet, device)
    rnet_detect = RNet_Detect(rnet, device)
    onet_detect = ONet_Detect(onet, device)

    _, boxes_align = pnet_detect(img_path)
    if boxes_align is None:
        return

    _, boxes_align = rnet_detect(img_path, boxes_align)
    if boxes_align is None:
        return

    _, boxes_align = onet_detect(img_path, boxes_align)
    if boxes_align is None:
        return

    img = cv2.imread(img_path)
    for i in enumerate(boxes_align.shape[0]):
        # 画人脸框
        cv2.rectangle(img, (boxes_align[i, 0], boxes_align[i, 1]), (boxes_align[i, 2], boxes_align[i, 3]), (255, 0, 0), 1)
        cv2.putText(img,'{:.2f}'.format(boxes_align[i, 4]), (boxes_align[i, 0], boxes_align[i, 1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('', 'model/pnet.pth', 'model/rnet.pth', 'model/onet.pth')


