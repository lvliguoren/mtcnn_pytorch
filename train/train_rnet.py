from model.mtcnn import RNet, ClS_Loss, Reg_Loss
from data.dataset import WiderDataSet
from torch.utils.data import DataLoader
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnet_dataset = WiderDataSet(data_dir='E:/TEST/MTCNN_DATA/24', pos_path='pos_24.txt', part_path='part_24.txt', neg_path='neg_24.txt')
rnet_dataloader = DataLoader(rnet_dataset, batch_size=512, shuffle=True)

rnet = RNet().to(device)
optimizer = torch.optim.Adam(rnet.parameters(), lr=1e-3)
critetion_cls = ClS_Loss(device)
critetion_reg = Reg_Loss(device)

def main():
    rnet.train()
    for batch_idx, (img, label, bbox_target, landmark_target) in enumerate(rnet_dataloader):
        img = img.to(device)
        label = label.to(device)
        bbox_target = bbox_target.to(device)

        pred_cls, pred_reg, pred_cls_pro = rnet(img)
        cls_loss, accuracy, precision, recall = critetion_cls(pred_cls, label, pred_cls_pro)
        reg_loss = critetion_reg(pred_reg, label, bbox_target)

        loss = cls_loss + reg_loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('batch_idx:{}, cls_loss:{:.4f}, reg_loss{:.4f}, loss:{:.4f}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}'
              .format(batch_idx, cls_loss, reg_loss, loss, accuracy, precision, recall))

        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.save(rnet, '../model/rnet.pth')

    print('train rnet complete')


if __name__ == '__main__':
    main()