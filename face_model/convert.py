
import enum
from pandas.io import json
import torch
from torch.backends import cudnn
import os
from threading import Thread
import cv2
import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset
from mobilenetV2 import mobilenetv2
from utils import save_checkpoint
from torch import nn
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import imageio
from datetime import datetime
import wandb
import onnx
# from onnx_tf.backend import prepare


class config:
    X_RES = 224
    Y_RES = 224
    POINTS = 20


def enable_gradient_layer(layer):
    layer.enable_gradient=False
    return layer

def main():
    model = mobilenetv2(num_classes=config.POINTS*2)
    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    num_face_blendshapes = 37
    linear = torch.nn.Linear(1280, num_face_blendshapes)
    newmodel = torch.nn.Sequential(*([torch.nn.ReLU()]  + list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1])) + [torch.nn.Flatten(), torch.nn.ReLU(), linear]))
    newmodel.cuda()
    newmodel = torch.nn.DataParallel(newmodel).cuda()
    checkpoint = torch.load('trained_models\\FuturaFaceBlenshapes\\2022-01-07_12-26-52-801156\\checkpoint-26.pth.tar')
    newmodel.load_state_dict(checkpoint['state_dict'])


    dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
    torch.onnx.export(newmodel.module, dummy_input, "model.onnx", verbose=True)

    # model = onnx.load('mnist.onnx')
    # tf_rep = prepare(model) 


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
    main()