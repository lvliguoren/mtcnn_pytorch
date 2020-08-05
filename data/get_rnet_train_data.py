import os
import torch
import numpy as np
from model.mtcnn import PNet_Detect
from data.utils import *
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(data_dir, model_path):
    anno_file = os.path.join(data_dir, 'wider_face_train.txt')
    image_dir = os.path.join(data_dir, 'WIDER_train/images')
    neg_save_dir = os.path.join(data_dir, "24/negative")
    pos_save_dir = os.path.join(data_dir, "24/positive")
    part_save_dir = os.path.join(data_dir, "24/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    f1 = open(os.path.join(data_dir, '24/pos_24.txt'), 'w')
    f2 = open(os.path.join(data_dir, '24/neg_24.txt'), 'w')
    f3 = open(os.path.join(data_dir, '24/part_24.txt'), 'w')

    with open(anno_file) as f:
        annotations = f.readlines()

    n_idx = 0
    p_idx = 0
    d_idx = 0
    for batch_idx, annotation in enumerate(annotations):
        annotation = annotation.strip().split(' ')
        im_path = os.path.join(image_dir, annotation[0] + '.jpg')
        img = cv2.imread(im_path)
        bbox_target = np.array(annotation[1:]).astype(np.float32)
        bbox_target = bbox_target.reshape(-1, 4)

        pnet = torch.load(model_path)
        pnet_detect = PNet_Detect(pnet, device)
        _, boxes_align = pnet_detect(im_path)

        if boxes_align is None:
            continue

        # 变成正方形
        boxes_align = convert_to_square(boxes_align)
        boxes_align[:, 0:4] = np.round(boxes_align[:, 0:4])

        cur_n_idx = 0
        for box in boxes_align:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left
            height = y_bottom - y_top
            # ignore box that is too small or beyond image border
            if width < 20 or x_left <= 0 or y_top <= 0 or x_right >= img.shape[1] or y_bottom >= img.shape[0]:
                continue
            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, bbox_target)
            cropped_im = img[y_top:y_bottom, x_left:x_right, :]
            resized_im = cv2.resize(cropped_im, (24, 24),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                cur_n_idx += 1
                if cur_n_idx <= 50:
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = bbox_target[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
        print('generate rnet img:{}/{}'.format(batch_idx + 1 ,len(annotations)))
    f1.close()
    f2.close()
    f3.close()


if __name__ == '__main__':
    main('E:/TEST/MTCNN_DATA/', '../model/pnet.pth')