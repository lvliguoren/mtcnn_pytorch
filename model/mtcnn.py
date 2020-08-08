import torch.nn as nn
import torch
import cv2
from torchvision import transforms
import numpy as np
from data.utils import convert_to_square


mean_vals = [0.471, 0.448, 0.408]
std_vals = [0.234, 0.239, 0.242]

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class PNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1 = BasicConv(3, 10, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = BasicConv(10, 16, kernel_size=3)
        self.conv3 = BasicConv(16, 32, kernel_size=3)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)  #N,C,H,W
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        cls = self.conv4_1(out)
        cls_pro = self.softmax4_1(cls) #cls
        reg = self.conv4_2(out) #box

        #预测时是单张图片 cls(1*2*H*W) reg(1*4*H*W) cls_pro(1*2*H*W)
        return cls, reg, cls_pro

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv(3, 28, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = BasicConv(28, 48, kernel_size=3)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = BasicConv(48, 64, kernel_size=2)
        self.linear1 = nn.Linear(576, 128)
        self.relu = nn.PReLU(128)
        self.linear2_1 = nn.Linear(128, 2)
        self.softmax2_1 = nn.Softmax(dim=1)
        self.linear2_2 = nn.Linear(128, 4)

    def forward(self, x):
        out = self.conv1(x) # N,C,H,W
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)

        out = out.permute(0, 2, 3, 1).contiguous() #N,H,W,C
        N, H, W, C = out.shape #1,3,3,64
        out = out.view(N, -1)
        out = self.linear1(out)
        out = self.relu(out)
        cls = self.linear2_1(out)
        cls_pro = self.softmax2_1(cls) #cls
        reg = self.linear2_2(out) #box

        return cls, reg, cls_pro

class ONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = BasicConv(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = BasicConv(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = BasicConv(64, 128, kernel_size=2)
        self.linear1 = nn.Linear(1152, 256)
        self.relu = nn.PReLU(256)
        self.linear2_1 = nn.Linear(256, 2)
        self.softmax2_1 = nn.Softmax(dim=1)
        self.linear2_2 = nn.Linear(256, 4)
        self.linear2_3 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.conv1(x) # N,C,H,W
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)

        out = out.permute(0, 2, 3, 1).contiguous() # N,H,W,C
        N,H,W,C = out.shape
        out = out.view(N, -1)
        out = self.linear1(out)
        out = self.relu(out)
        cls = self.linear2_1(out)
        cls_pro = self.softmax2_1(cls) #cls
        reg = self.linear2_2(out) #box
        landmark = self.linear2_3(out) #point

        return cls, reg, cls_pro, landmark

class ClS_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.critetion = nn.CrossEntropyLoss().to(device)

    def forward(self, pred, target, pred_pro):
        #训练的时候是批量训练12*12的图片(图片类型分别为1，0，-1)
        pred = pred.squeeze(dim=-1).squeeze(dim=-1) # (N*2*1*1) -> N*2
        pred_pro = pred_pro.squeeze(dim=-1).squeeze(dim=-1) # (N*2*1*1) -> N*2
        _, pred_pro = pred_pro.max(1)
        # 选出类型为0和1的样本做分类
        selected_idx = (target != -1).nonzero()[:,0]
        if selected_idx.numel() > 0:
            cls_loss = self.critetion(pred[selected_idx], target[selected_idx])
        else:
            cls_loss = torch.tensor(0.0).to(self.device)

        label_one = (target == 1).nonzero()[:,0]
        label_zero = (target == 0).nonzero()[:,0]
        pred_one = (pred_pro == 1).nonzero()[:,0]
        pred_zero = (pred_pro == 0).nonzero()[:,0]

        tp = [i for i in pred_one if i in label_one]
        tn = [i for i in pred_zero if i in label_zero]
        fp = [i for i in pred_one if i in label_zero]
        fn = [i for i in pred_zero if i in label_one]

        accuracy = (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))
        if len(tp) + len(fp) == 0:
            precision = torch.tensor(0.0).to(self.device)
            recall = torch.tensor(0.0).to(self.device)
        else:
            precision = len(tp) / (len(tp) + len(fp))
            recall = len(tp) / (len(tp) + len(fn))

        return cls_loss, accuracy, precision, recall

class Reg_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.critetion = nn.SmoothL1Loss().to(device)

    def forward(self, pred, label, target):
        #训练的时候是批量训练12*12的图片(图片类型分别为1，0，-1)
        pred = pred.squeeze(dim=-1).squeeze(dim=-1)  # (N*4*1*1) -> N*4

        # 选出类型为-1和1的样本做回归
        selected_idx = (abs(label) == 1).nonzero()[:,0]
        if selected_idx.numel() > 0:
            reg_loss = self.critetion(pred[selected_idx], target[selected_idx])
        else:
            reg_loss = torch.tensor(0.0).to(self.device)

        return reg_loss

class PNet_Detect(nn.Module):
    def __init__(self, pnet, device):
        super().__init__()
        self.pnet = pnet.to(device)
        self.device = device

    def forward(self, img_path):
        net_size = 12
        min_face_size = 20
        current_scale = float(net_size) / min_face_size
        im_resized = self.processed_image(img_path, current_scale)
        _, current_height, current_width = im_resized.shape
        im_resized = im_resized.unsqueeze(dim=0)
        all_boxes = list()
        # 图像金字塔
        while min(current_height, current_width) > net_size:
            # 类别和box
            im_resized = im_resized.to(self.device)
            prd_cls, pred_reg, pred_cls_pro = self.pnet(im_resized)
            pred_cls_pro = pred_cls_pro.squeeze()
            pred_cls_pro = pred_cls_pro.permute(1, 2, 0).cpu().detach().numpy() # H,W,C
            pred_reg = pred_reg.squeeze()
            pred_reg = pred_reg.permute(1, 2, 0).cpu().detach().numpy() # H,W,C
            boxes = self.generate_bbox(pred_cls_pro[:, :, 1], pred_reg, current_scale, 0.6)
            current_scale *= 0.79  # 继续缩小图像做金字塔
            im_resized = self.processed_image(img_path, current_scale)
            _, current_height, current_width = im_resized.shape
            im_resized = im_resized.unsqueeze(dim=0)

            if boxes.size == 0:
                continue

            # 非极大值抑制留下重复低的box
            keep = nms(boxes[:, :5], 0.5)
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes)==0:
            return None,None

        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        keep = nms(all_boxes[:, 0:5], 0.7)
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.transpose()

        return boxes, boxes_c


    def processed_image(self, img_path, scale):
        '''预处理数据，转化图像尺度并对像素归一到[-1,1]
        '''
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        img = transforms.ToTensor()(img_resized)
        img = transforms.Normalize(mean=mean_vals, std=std_vals)(img)
        return img

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
         得到对应原图的box坐标，分类分数，box偏移量
        """
        # pnet大致将图像size缩小2倍
        stride = 2

        cellsize = 12

        # 将置信度高的留下
        t_index = np.where(cls_map > threshold)

        # 没有人脸
        if t_index[0].size == 0:
            return np.array([])
        # 偏移量
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        # 对应原图的box坐标，分类分数，box偏移量
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        # shape[n,9]
        return boundingbox.transpose()

class RNet_Detect(nn.Module):
    def __init__(self, rnet, device):
        super().__init__()
        self.rnet = rnet.to(device)
        self.device = device

    def forward(self, img_path, dets):
        img = cv2.imread(img_path)
        h, w, c = img.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            if tmph[i] > 0 and tmpw[i] > 0:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                crop_im = cv2.resize(tmp, (24, 24))
                crop_im_tensor = transforms.ToTensor()(crop_im)
                crop_im_tensor = transforms.Normalize(mean=mean_vals, std=std_vals)(crop_im_tensor)
                # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
                cropped_ims_tensors.append(crop_im_tensor)

        if len(cropped_ims_tensors) == 0:
            return None,None

        feed_imgs = torch.stack(cropped_ims_tensors).to(self.device)

        # 生成的大批量数据会导致显存溢出，整理为小批量数据预测
        minibatch = []
        minibatch_size = 50
        pred_cls_arr = []
        pred_reg_arr = []
        pred_cls_pro_arr = []
        cur = 0
        n = feed_imgs.shape[0]
        while cur < n:
            minibatch.append(feed_imgs[cur:min(cur + minibatch_size, n),:,:,:])
            cur += minibatch_size

        for imgs in minibatch:
            pred_cls, pred_reg, pred_cls_pro = self.rnet(imgs)
            pred_cls = pred_cls.cpu().detach().numpy()
            pred_cls_pro = pred_cls_pro.cpu().detach().numpy()
            pred_reg = pred_reg.cpu().detach().numpy()

            pred_cls_arr.append(pred_cls)
            pred_reg_arr.append(pred_reg)
            pred_cls_pro_arr.append(pred_cls_pro)

        # list拼接为nparray类型
        pred_cls_np = np.concatenate(pred_cls_arr, axis=0)
        pred_reg_np = np.concatenate(pred_reg_arr, axis=0)
        pred_cls_pro_np = np.concatenate(pred_cls_pro_arr, axis=0)
        keep = np.where(pred_cls_pro_np[:, 1] > 0.7)[0]

        if len(keep) == 0:
            return None,None

        dets = dets[keep]
        pred_cls_keep = pred_cls_np[keep]
        pred_cls_pro_keep = pred_cls_pro_np[keep]
        pred_reg_keep = pred_reg_np[keep]

        keep = nms(dets, 0.7)
        if len(keep) == 0:
            return None, None

        dets = dets[keep]
        pred_cls_keep = pred_cls_keep[keep]
        pred_cls_pro_keep = pred_cls_pro_keep[keep]
        pred_reg_keep = pred_reg_keep[keep]

        # box的长宽
        bbw = dets[:, 2] - dets[:, 0] + 1
        bbh = dets[:, 3] - dets[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([dets[:, 0] + pred_reg_keep[:, 0] * bbw,
                             dets[:, 1] + pred_reg_keep[:, 1] * bbh,
                             dets[:, 2] + pred_reg_keep[:, 2]* bbw,
                             dets[:, 3] + pred_reg_keep[:, 3] * bbh,
                             pred_cls_pro_keep[:, 1]])
        boxes_c = boxes_c.transpose()
        return dets, boxes_c

class ONet_Detect(nn.Module):
    def __init__(self, onet, device):
        super().__init__()
        self.onet = onet.to(device)
        self.device = device

    def forward(self, img_path, dets):
        img = cv2.imread(img_path)
        h, w, c = img.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            if tmph[i] > 0 and tmpw[i] > 0:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                crop_im = cv2.resize(tmp, (48, 48))
                crop_im_tensor = transforms.ToTensor()(crop_im)
                crop_im_tensor = transforms.Normalize(mean=mean_vals, std=std_vals)(crop_im_tensor)
                # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
                cropped_ims_tensors.append(crop_im_tensor)

        if len(cropped_ims_tensors) == 0:
            return None,None

        feed_imgs = torch.stack(cropped_ims_tensors).to(self.device)

        # 生成的大批量数据会导致显存溢出，整理为小批量数据预测
        minibatch = []
        minibatch_size = 50
        pred_cls_arr = []
        pred_reg_arr = []
        pred_cls_pro_arr = []
        cur = 0
        n = feed_imgs.shape[0]
        while cur < n:
            minibatch.append(feed_imgs[cur:min(cur + minibatch_size, n),:,:,:])
            cur += minibatch_size

        for imgs in minibatch:
            pred_cls, pred_reg, pred_cls_pro, landmark = self.onet(imgs)
            pred_cls = pred_cls.cpu().detach().numpy()
            pred_cls_pro = pred_cls_pro.cpu().detach().numpy()
            pred_reg = pred_reg.cpu().detach().numpy()

            pred_cls_arr.append(pred_cls)
            pred_reg_arr.append(pred_reg)
            pred_cls_pro_arr.append(pred_cls_pro)

        # list拼接为nparray类型
        pred_cls_np = np.concatenate(pred_cls_arr, axis=0)
        pred_reg_np = np.concatenate(pred_reg_arr, axis=0)
        pred_cls_pro_np = np.concatenate(pred_cls_pro_arr, axis=0)
        keep = np.where(pred_cls_pro_np[:, 1] > 0.7)[0]

        if len(keep) == 0:
            return None,None

        dets = dets[keep]
        pred_cls_keep = pred_cls_np[keep]
        pred_cls_pro_keep = pred_cls_pro_np[keep]
        pred_reg_keep = pred_reg_np[keep]

        keep = nms(dets, 0.3)
        if len(keep) == 0:
            return None, None

        dets = dets[keep]
        pred_cls_keep = pred_cls_keep[keep]
        pred_cls_pro_keep = pred_cls_pro_keep[keep]
        pred_reg_keep = pred_reg_keep[keep]

        # box的长宽
        bbw = dets[:, 2] - dets[:, 0] + 1
        bbh = dets[:, 3] - dets[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([dets[:, 0] + pred_reg_keep[:, 0] * bbw,
                             dets[:, 1] + pred_reg_keep[:, 1] * bbh,
                             dets[:, 2] + pred_reg_keep[:, 2]* bbw,
                             dets[:, 3] + pred_reg_keep[:, 3] * bbh,
                             pred_cls_pro_keep[:, 1]])
        boxes_c = boxes_c.transpose()
        return dets, boxes_c

def pad(bboxes, w, h):
            """
                pad the the boxes
            Parameters:
            ----------
                bboxes: numpy array, n x 5, input bboxes
                w: float number, width of the input image
                h: float number, height of the input image
            Returns :
            ------
                dy, dx : numpy array, n x 1, start point of the bbox in target image
                edy, edx : numpy array, n x 1, end point of the bbox in target image
                y, x : numpy array, n x 1, start point of the bbox in original image
                ey, ex : numpy array, n x 1, end point of the bbox in original image
                tmph, tmpw: numpy array, n x 1, height and width of the bbox
            """

            tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
            tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
            numbox = bboxes.shape[0]

            dx = np.zeros((numbox,))
            dy = np.zeros((numbox,))
            edx, edy = tmpw.copy() - 1, tmph.copy() - 1

            x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

            tmp_index = np.where(ex > w - 1)
            if len(tmp_index[0]) > 0:
                edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
                ex[tmp_index] = w - 1

            tmp_index = np.where(ey > h - 1)
            if len(tmp_index[0]) > 0:
                edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
                ey[tmp_index] = h - 1

            tmp_index = np.where(x < 0)
            if len(tmp_index[0]) > 0:
                dx[tmp_index] = 0 - x[tmp_index]
                x[tmp_index] = 0

            tmp_index = np.where(y < 0)
            if len(tmp_index[0]) > 0:
                dy[tmp_index] = 0 - y[tmp_index]
                y[tmp_index] = 0

            return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
            return_list = [item.astype(np.int32) for item in return_list]

            return return_list

def nms(dets, thresh):
        '''剔除太相似的box'''
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 将概率值从大到小排列
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

            # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
