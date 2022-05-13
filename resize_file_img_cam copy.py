import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import glob

import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import cv2
from pandas import DataFrame #엑셀저장

from network_CNN import *

#________________________gpu사용여부
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
#___________________________________

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


cam_net = CNN().to(device)
cam_net.load_state_dict(torch.load('weight/black_img/class4/model_c4_(5e-05)_(124)_(95.29).pth'))

#___________________________dataload
#Tensor로 이미지 변환
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])


path = glob.glob("image/raw_img/*.jpg")
count = 1
for link in path:
    print('____________')
    one_img = Image.open(link)
    i_w, i_h = one_img.size
    test_set = trans(one_img)
    print(link)
    count +=1
    #___________________________________

    with torch.no_grad():
        imgs = test_set
        print(imgs.shape)
        imgs = imgs.unsqueeze(0).to(device)
        #print(imgs.shape)
        prediction,f = cam_net(imgs)
        
        #label과 prediction 값이 일치하면 1 아니면 0
        correct_prediction = torch.argmax(prediction,1)
        
    classes =  ('ground','obstacle','raw','sky')
    params = list(cam_net.parameters())
    #print(params[0])
    num = 0

    print("ANS :",classes[int(correct_prediction)]," REAL :",classes[0],num)

    #print(f.shape)
    overlay = params[-2][2].matmul(f.reshape(512,196)).reshape(14,14).cpu().data.numpy()
    #overlay = overlay - np.min(overlay)
    #ground = params[-2][0].matmul(f.reshape(512,49)).reshape(7,7).cpu().data.numpy()
    #obstacle = params[-2][1].matmul(f.reshape(512,49)).reshape(7,7).cpu().data.numpy()# - obstacle - tree
    #overlay = overlay - ground - obstacle
    overlay = overlay - np.min(overlay)
    #overlay = overlay / np.max(overlay)
    print('최대값은 ', np.max(overlay))

    #___________________save data
    data = DataFrame(overlay)
    #print(data)
    data.to_excel('CAM_data.xlsx')

    #print(imgs.cpu().shape)
    #imshow(imgs[0].cpu())
    plt.imshow(one_img)
    plt.imshow(skimage.transform.resize(overlay, [i_h ,i_w]), alpha=0.6,cmap='jet')
    plt.show()