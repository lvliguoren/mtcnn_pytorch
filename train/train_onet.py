from model.mtcnn import ONet, ClS_Loss, Reg_Loss
from data.dataset import WiderDataSet
from torch.utils.data import DataLoader
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
onet_dataset = WiderDataSet(data_dir='E:/TEST/MTCNN_DATA/48', pos_path='pos_48.txt', part_path='part_48.txt', neg_path='neg_48.txt')
onet_dataloader = DataLoader(onet_dataset, batch_size=512, shuffle=True)

onet = ONet().to(device)
optimizer = torch.optim.Adam(onet.parameters(), lr=1e-3)
critetion_cls = ClS_Loss(device)
critetion_reg = Reg_Loss(device)

def main():
    onet.train()
    for batch_idx, (img, label, bbox_target, landmark_target) in enumerate(onet_dataloader):
        img = img.to(device)
        label = label.to(device)
        bbox_target = bbox_target.to(device)

        pred_cls, pred_reg, pred_cls_pro, _ = onet(img)
        cls_loss, accuracy, precision, recall = critetion_cls(pred_cls, label, pred_cls_pro)
        reg_loss = critetion_reg(pred_reg, label, bbox_target)

        loss = cls_loss + reg_loss * 1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('batch_idx:{}, cls_loss:{:.4f}, reg_loss{:.4f}, loss:{:.4f}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}'
              .format(batch_idx, cls_loss, reg_loss, loss, accuracy, precision, recall))

        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.save(onet, '../model/onet.pth')

    print('train onet complete')


if __name__ == '__main__':
    main()