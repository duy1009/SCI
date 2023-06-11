import os
# import sys
import numpy as np
import torch
# import argparse
import torch.utils
# import torch.backends.cudnn as cudnn
# from PIL import Image
# from torch.autograd import Variable
from model import Finetunemodel
import time
# from multi_read_data import MemoryFriendlyLoader
import cv2
# parser = argparse.ArgumentParser("SCI")



# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--seed', type=int, default=2, help='random seed')

# args = parser.parse_args()
# save_path = "./results/medium"
data_path = "./data/medium/3020.jpg"
# data_path = "./data/medium"
model_path = 'C:\\Users\\admin\\Desktop\\SCI\\Model\\2100-Image\\weights_0.pt'
# os.makedirs(save_path, exist_ok=True)


def cv2_to_torch(img):  #img is a 3D tensor
    img_norm = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_norm.astype(np.float32)/255
    img_norm = np.transpose(img_norm, (2, 0, 1))   
    return torch.from_numpy(img_norm).unsqueeze(0) #tensor is a 4D tensor

def torch_to_cv2(tensor): #tensor is a 4D tensor
    img = np.transpose(tensor[0]*255, (1, 2, 0)).numpy().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # return img

def main():
    
    model = Finetunemodel(model_path)

    model.eval()
    while True:
        with torch.no_grad():

            img_low = cv2.imread(data_path)
            input = cv2_to_torch(img_low)
            print(input.shape)
            i,r = model(input)
            img_high = torch_to_cv2(r)
            cv2.imshow('image', img_high)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
