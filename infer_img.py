#!/usr/bin/env python
# -*- coding:utf-8-*-
# file: infer_img.py
# @author: jory.d
# @contact: dangxusheng163@163.com
# @time: 2022/6/10 16:18
# @desc: 

import os, os.path as osp
import glob
import numpy as np
import cv2
from PIL import Image
from skimage import io, transform, color
import torch
from torchvision import transforms
from model import BASNet
from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab

"""
conda activate py37
cd E:/Projects/PycharmProjects/BASNet-master
python infer_img.py
"""


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def save_output(image_name, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    # print(np.max(predict_np), np.min(predict_np))

    # get min bounding box
    mask_h, mask_w = np.where(predict_np > 0.6)
    x1 = np.min(mask_w)
    x2 = np.max(mask_w)
    y1 = np.min(mask_h)
    y2 = np.max(mask_h)
    mask_box = [x1, y1, x2, y2]

    im = Image.fromarray(predict_np * 255).convert('RGB')
    # imo = im
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    scale_hw = np.asarray(predict_np.shape) *1.0/ np.asarray(image.shape[:2])
    scale_wh = list(scale_hw)[::-1]*2
    mask_box = np.asarray(mask_box) / np.asarray(scale_wh)
    mask_box = list(map(int, mask_box))
    print(imo.size, mask_box)

    cv_img = np.array(imo, dtype=np.uint8)
    cv_img[:, :, 0][cv_img[:, :, 0] > 0] = 255
    cv_img[:, :, 1][cv_img[:, :, 1] > 0] = 255
    cv_img[:, :, 2][cv_img[:, :, 2] > 0] = 0
    cv2.addWeighted(image, 0.8, cv_img, 0.2, 0, dst=cv_img)
    cv2.rectangle(cv_img, tuple(mask_box[:2]), tuple(mask_box[2:]), (0, 255, 0), 1)

    filename = osp.basename(image_name)

    _savefile = d_dir + filename + '.png'
    print(_savefile)
    os.makedirs(osp.dirname(_savefile), exist_ok=True)
    cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR, dst=cv_img)
    cv2.imwrite(_savefile, cv_img)
    # imo.save(_savefile)


@torch.no_grad()
def test_imgs():
    # image_dir = 'E:/Baidu_download/baidu_download_20220531/images/test_imgMatting/'
    # prediction_dir = 'E:/Baidu_download/baidu_download_20220531/images/test_imgMatting_results/'
    # image_dir = 'D:/Datasets/Tray_Data/20220622_114400'
    # prediction_dir = 'D:/Datasets/Tray_Data/20220622_114400' + '_results/'

    # image_dir = '/media/sunny-dxs/Workspace2/Datasets/spider_imgs/tray/塑料平板托盘-20220117'
    image_dir = '/media/sunny-dxs/Workspace2/Datasets/spider_imgs/tray/塑料网格托盘-20220117'
    # image_dir = '/media/sunny-dxs/Workspace2/Datasets/spider_imgs/tray/田字托盘-20220117'
    prediction_dir = f'{image_dir}/../' + 'imgMatting_results/塑料网格托盘-20220117'

    model_dir = './ckpt/basnet.pth'

    assert osp.exists(model_dir)

    img_files = []
    for i, f in enumerate(glob.iglob(f'{image_dir}/**/*.jpg', recursive=True)):
        # if i >500: break
        img_files.append(f)


    assert len(img_files) > 0
    sorted(img_files)

    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    transform = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])

    # --------- 4. inference for each image ---------
    for f in img_files:
        image = io.imread(f)
        label_3 = np.zeros(image.shape)

        # preprocess
        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        sample = {'image': image, 'label': label}

        data_test = transform(sample)
        assert isinstance(data_test, dict)
        print(data_test['image'].shape)

        inputs_test = data_test['image']
        inputs_test = inputs_test[None, ...]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(f, pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7, d8

    print('done.')


if __name__ == '__main__':
    test_imgs()
