
import torch
import os
from threading import Thread
import cv2
import numpy as np
from skimage import io

from torch import nn
from mobilenetV2 import mobilenetv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
import imageio


class config:
    X_RES = 224
    Y_RES = 224
    POINTS = 20

shapeKeys = [
    'JawRight',
    'JawLeft',
    'JawForward',
    'JawOpen',
    'MouthApeShape',
    'MouthUpperRight',
    'MouthUpperLeft',
    'MouthLowerRight',
    'MouthLowerLeft',
    'MouthUpperOverturn',
    'MouthLowerOverturn',
    'MouthPout',
    'MouthSmileRight',
    'MouthSmileLeft',
    'MouthSadRight',
    'MouthSadLeft',
    'CheekPuffRight',
    'CheekPuffLeft',
    'CheekSuck',
    'MouthUpperUpRight',
    'MouthUpperUpLeft',
    'MouthLowerDownRight',
    'MouthLowerDownLeft',
    'MouthUpperInside',
    'MouthLowerInside',
    'MouthLowerOverlay',
    'TongueLongStep1',
    'TongueLongStep2',
    'TongueDown',
    'TongueUp',
    'TongueRight',
    'TongueLeft',
    'TongueRoll',
    'TongueUpLeftMorph',
    'TongueUpRightMorph',
    'TongueDownLeftMorph',
    'TongueDownRightMorph',
]


def log_prediction(step, input, output, target):
    shapeKeys = [
        'JawRight',
        'JawLeft',
        'JawForward',
        'JawOpen',
        'MouthApeShape',
        'MouthUpperRight',
        'MouthUpperLeft',
        'MouthLowerRight',
        'MouthLowerLeft',
        'MouthUpperOverturn',
        'MouthLowerOverturn',
        'MouthPout',
        'MouthSmileRight',
        'MouthSmileLeft',
        'MouthSadRight',
        'MouthSadLeft',
        'CheekPuffRight',
        'CheekPuffLeft',
        'CheekSuck',
        'MouthUpperUpRight',
        'MouthUpperUpLeft',
        'MouthLowerDownRight',
        'MouthLowerDownLeft',
        'MouthUpperInside',
        'MouthLowerInside',
        'MouthLowerOverlay',
        'TongueLongStep1',
        'TongueLongStep2',
        'TongueDown',
        'TongueUp',
        'TongueRight',
        'TongueLeft',
        'TongueRoll',
        'TongueUpLeftMorph',
        'TongueUpRightMorph',
        'TongueDownLeftMorph',
        'TongueDownRightMorph',
    ]

    # img = input.cpu().numpy()[0]
    # img = np.transpose(img, (2, 1, 0)).copy() * 255
    fig = Figure(figsize=(10, 10), dpi=60)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0,0,1,1])

    labels = shapeKeys
    out_l =  output.cpu().detach().numpy()[0]
    target_l = target.cpu().numpy()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    ax.bar(x - width/2, target_l, width, label='Target')
    ax.bar(x + width/2, out_l, width, label='Output')
    ax.autoscale(enable=None, axis="x", tight=True)
    ax.autoscale(enable=None, axis="y", tight=True)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Value')
    ax.set_title('Prediction against target')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    pred = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    return pred

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


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
        img = self.frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = grayConversion(img)
        # print(self.frame.shape)
        img = cv2.merge([img,img,img]).astype(dtype=np.uint8)
        img = cv2.resize(img, dsize=(config.X_RES, config.Y_RES))
        tensor = torch.from_numpy(np.transpose(img / 255, (2, 1, 0))).float().unsqueeze(0)
     

        # data_img = np.reshape(img, (config.X_RES, config.Y_RES, 1))
        # print(img.shape)
        # tensor = torch.from_numpy(np.transpose(img / 255, (2, 1, 0))).float()
      
        # print(tensor.shape)
        res = self.model(tensor)
        print(res.cpu().detach().numpy()[0][3])

        # fig = Figure(figsize=(2, 2), dpi=100)
        # canvas = FigureCanvasAgg(fig)
        # ax = fig.add_axes([0,0,1,1])

        # labels = shapeKeys
        # out_l = res.cpu().detach().numpy()[0]
        # # target_l = target.cpu().numpy()[0]

        # x = np.arange(len(labels))  # the label locations
        # width = 0.35  # the width of the bars

        # # ax.bar(x - width/2, target_l, width, label='Target')
        # ax.bar(x + width/2, out_l, width, label='Output')

        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Value')
        # ax.set_title('Prediction')
        # ax.set_xticks(x, labels)
        # ax.legend()

        # fig.canvas.draw()

        # s, (width, height) = canvas.print_to_buffer()


        # res = res.reshape((68, 2))
        # for e in range(68):
        #     cv2.circle(img, (int(res[e][0] * config.X_RES) , int(res[e][1] * config.Y_RES)) , 1, (255, 0, 0), 2)
        # points = model.predict(images)[0]
        # for i in range(int(len(res) / 2)):
        #     cv2.circle(img, (int(res[i * 2] * config.X_RES) , int(res[i * 2 + 1] * config.Y_RES)) , 1, (255, 0, 0), 2)
        cv2.imshow('frame', img)
        # cv2.imshow('pred', np.frombuffer(s, np.uint8).reshape((height, width, 4)))
        cv2.waitKey(1)

def enable_gradient_layer(layer):
    layer.enable_gradient=False
    return layer


def main():
   
    num_face_blendshapes = 37
    model = mobilenetv2(num_classes=config.POINTS*2)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    # checkpoint = torch.load('trained_models\\Mobilenet-ffhq-croped\\2021-12-16_23-48-32-851676\\checkpoint-40.pth.tar')
    # model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    linear = torch.nn.Linear(1280, num_face_blendshapes)
    newmodel = torch.nn.Sequential(*([torch.nn.ReLU()]  + list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1])) + [torch.nn.Flatten(), torch.nn.ReLU(), linear]))
    # newmodel = model
    # newmodel = torch.nn.Sequential(*(list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1]))))
    newmodel.cuda()
    newmodel = torch.nn.DataParallel(newmodel).cuda()
    checkpoint = torch.load('trained_models\\FuturaFaceBlenshapes\\2022-01-07_02-44-06-318806\\checkpoint-59.pth.tar')
    newmodel.load_state_dict(checkpoint['state_dict'])

    dataset_folder = 'datasets/recorded_dataset_1641092589206/'
    dataset_file = dataset_folder + 'model.json'
    dataset = pd.read_json(dataset_file)

    for record_name, b in dataset.iterrows():
        record_path = dataset_folder + record_name + '.mp4'
        shape = b['blendshapes']['shape']
        reader = imageio.get_reader(record_path)
        for frame_number, im in enumerate(reader):
            img = grayConversion(im)
            img = cv2.merge([img,img,img]).astype(dtype=np.uint8)
            img = cv2.resize(img, dsize=(config.X_RES, config.Y_RES))
            target = torch.from_numpy(np.array(shape).astype(dtype=np.float32)).float()
            tensor = torch.from_numpy(np.transpose(img / 255, (2, 1, 0))).float().unsqueeze(0)
            res = newmodel(tensor)
            (pred) = log_prediction(0, tensor, res, target)
            cv2.imshow('frame', pred)
            cv2.waitKey(1)
            break

    # src = 'http://192.168.1.28:81/stream'
    # threaded_camera = ThreadedCamera(src, newmodel)
    # while True:
    #     try:
    #         threaded_camera.show_frame()
    #     except AttributeError:
    #         pass



if __name__ == '__main__':
    main()