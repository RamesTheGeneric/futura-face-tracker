
from pandas.io import json
import torch
import os
from threading import Thread
import cv2
import numpy as np
from skimage import io
from torch.utils.data.dataset import Dataset
from mobilenetV2 import mobilenetv2
from torch import nn
import pandas as pd
from torch.utils.data.dataloader import DataLoader


class config:
    X_RES = 224
    Y_RES = 224
    POINTS = 20

def enable_gradient_layer(layer):
    layer.enable_gradient=False
    return layer
    

class BlendshapesDataset(Dataset):
    def __init__(self, json_file, root_dir):
        self.frame = pd.read_json(json_file)
        self.root_dir = root_dir
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx,  self.frame.iloc[idx])

        return { 'image': 'hey' }

def main():
    model = mobilenetv2(num_classes=config.POINTS*2)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load('trained_models\\Mobilenet-ffhq-croped\\2021-12-16_23-48-32-851676\\checkpoint-40.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    num_face_blendshapes = 36
    linear = torch.nn.Linear(1280, num_face_blendshapes)

    newmodel = torch.nn.Sequential(*(list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1])) + [linear]))
    # print(newmodel)

    dataset_folder = 'datasets/recorded_dataset_1640050443080/'
    dataset_file = dataset_folder + 'model.json'

    dataset = BlendshapesDataset(
        json_file=dataset_file,
        root_dir=dataset_folder,
    )
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    
    epochs = 1

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            # print(batch)
            print('hey')
    
    # print(dataset_frame)

    

    # batch = torch.stack([ get_image() for e in range(64) ], dim=0).cuda()

    # for image_name, b in dataset_frame.iterrows():
    #     image_path = dataset_folder + image_name
    #     shape = b['blendshapes']['shape']



if __name__ == '__main__':
    main()