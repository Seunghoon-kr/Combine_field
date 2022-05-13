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


#___________________________dataload
#Tensor로 이미지 변환
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])

one_img = Image.open('image/raw_img/20201007_143942.jpg')
i_w, i_h = one_img.size
test_set = trans(one_img)

#___________________________________

cam_net = CNN().to(device)
path = glob.glob("weight/black_img/*.pth")
count = 1
for link in path:
    cam_net.load_state_dict(torch.load(link))
    print('____________')
    print(count)
    count += 1

    with torch.no_grad():
        imgs = test_set
        print(imgs.shape)
        imgs = imgs.unsqueeze(0).to(device)
        #print(imgs.shape)
        prediction,f = cam_net(imgs)
        
        #label과 prediction 값이 일치하면 1 아니면 0
        correct_prediction = torch.argmax(prediction,1)
        
    


    classes =  ('ground','obstacle','raw')
    params = list(cam_net.parameters())
    #print(params[0])
    num = 0

    print("ANS :",classes[int(correct_prediction)]," REAL :",classes[0],num)

    #print(params[-2][0].shape)
    #print(f.shape)
    overlay = params[-2][2].matmul(f.reshape(512,49)).reshape(7,7).cpu().data.numpy()
    #tree = params[-2][2].matmul(f.reshape(512,49)).reshape(7,7).cpu().data.numpy()
    #overlay = params[-2][0].matmul(f.reshape(512,49)).reshape(7,7).cpu().data.numpy()# - obstacle - tree
    overlay = overlay - np.min(overlay)
    #overlay = overlay / np.max(overlay)

    #___________________save data
    data = DataFrame(overlay)
    #print(data)
    data.to_excel('CAM_data.xlsx')

    #print(imgs.cpu().shape)
    #imshow(imgs[0].cpu())
    plt.imshow(one_img)
    plt.imshow(skimage.transform.resize(overlay, [i_h ,i_w]), alpha=0.6,cmap='jet')
    plt.show()