
import torch
import os
from threading import Thread
import cv2
import numpy as np
from skimage import io

from torch import nn
from mobilenetV2 import mobilenetv2

class config:
    X_RES = 224
    Y_RES = 224

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  

class ThreadedCamera(object):
    def __init__(self, src, model):
        self.model = model
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
       

    def show_frame(self):
        img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        # img = io.imread("C:\\Users\\louca\\Documents\\FuturaServer\\datasets\\recorded_dataset_1637092614433\img-885.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([img,img,img])
        img = cv2.resize(img, dsize=(config.X_RES, config.Y_RES))
        
        # data_img = np.reshape(img, (config.X_RES, config.Y_RES, 1))
        tensor = torch.from_numpy(np.transpose(img  / 255, (2, 1, 0)))
        tensor = tensor.float().unsqueeze(0)
        res = self.model(tensor)
        res = res.reshape((68, 2))
        for e in range(68):
            cv2.circle(img, (int(res[e][0] * config.X_RES) , int(res[e][1] * config.Y_RES)) , 1, (255, 0, 0), 2)
        # points = model.predict(images)[0]
        # for i in range(int(len(res) / 2)):
        #     cv2.circle(img, (int(res[i * 2] * config.X_RES) , int(res[i * 2 + 1] * config.Y_RES)) , 1, (255, 0, 0), 2)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

def main():
   
    model = mobilenetv2()

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load('trained_models\\Mobilenet-ffhq-croped\\2021-11-21_16-22-20-280997\\checkpoint-13.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # img = io.imread("C:\\Users\\louca\\Documents\\Futurabeast\\futura-face\\old\\Dataset\\ezgif-4-a7f56d5083de-jpg\\ezgif-frame-186.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.merge([img,img,img])
    # # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # # hsv[:,:,2] += 50
    # # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # img = cv2.resize(img, dsize=(config.X_RES, config.Y_RES))

    # # data_img = np.reshape(img, (config.X_RES, config.Y_RES, 1))
    # tensor = torch.from_numpy(np.transpose(img  / 255, (2, 1, 0)))
    # tensor = tensor.float().unsqueeze(0)
    # res = model(tensor)
    # res = res.reshape((68, 2))
    # for e in range(68):
    #     cv2.circle(img, (int(res[e][0] * config.X_RES) , int(res[e][1] * config.Y_RES)) , 1, (255, 0, 0), 2)
    # # points = model.predict(images)[0]
    # # for i in range(int(len(res) / 2)):
    # #     cv2.circle(img, (int(res[i * 2] * config.X_RES) , int(res[i * 2 + 1] * config.Y_RES)) , 1, (255, 0, 0), 2)
    # cv2.imshow('frame', img)
    # cv2.waitKey(0) 

    # image = io.imread("C:\\Users\\louca\\Documents\\Futurabeast\\futura-face\\datasets\\preprocessed_dataset\\images1024x1024\\04000\\04002.png")
    # tensor = torch.from_numpy(image)
    # tensor = torch.stack([tensor, tensor, tensor], dim=0).float().unsqueeze(0)
    # res = model(tensor)
    # res = res.reshape((2, 68))
    # for e in range(68):
    #     cv2.circle(image, (int(res[0][e] * config.X_RES) , int(res[1][e] * config.Y_RES)) , 1, (255, 0, 0), 2)
    # cv2.imshow('frame', image)
    src = 'http://192.168.1.13:81/stream'
    threaded_camera = ThreadedCamera(src, model)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass
    # prediction = model()



if __name__ == '__main__':
    main()