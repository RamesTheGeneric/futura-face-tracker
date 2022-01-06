
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


class config:
    X_RES = 224
    Y_RES = 224
    POINTS = 20

def enable_gradient_layer(layer):
    layer.enable_gradient=False
    return layer

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def recorded_shapes(dataset, dataset_folder, batch_size=40):
    for record_name, b in dataset.iterrows():
        record_path = dataset_folder + record_name + '.mp4'
        shape = b['blendshapes']['shape']
        reader = imageio.get_reader(record_path)
        inputs = []
        targets = []
        for frame_number, im in enumerate(reader):
            image = grayConversion(im)
            image = cv2.merge([image,image,image]).astype(dtype=np.float32)
            image = cv2.resize(image, dsize=(config.X_RES, config.Y_RES))
            image = image / 255
            targets.append(torch.from_numpy(np.array(shape)).float())
            inputs.append(torch.from_numpy(np.transpose(image, (2, 1, 0))).float())

            if (frame_number != 0 and frame_number == 10):
                stacked = torch.stack(inputs, dim=0).cuda()
                tstacked = torch.stack(targets, dim=0).cuda()
                inputs = []
                targets = []
                yield tstacked, stacked
                break

def main():

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

    wandb.init(project="FuturaFaceTacker", name="FuturaFaceBlenshapes-" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S-%f'))

    

    checkpoint_folder = 'trained_models/FuturaFaceBlenshapes/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')


    model = mobilenetv2(num_classes=config.POINTS*2)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load('trained_models\\Mobilenet-ffhq-croped\\2021-12-16_23-48-32-851676\\checkpoint-40.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    num_face_blendshapes = 37
    linear = torch.nn.Linear(1280, num_face_blendshapes)
    newmodel = torch.nn.Sequential(*(list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1])) + [torch.nn.Flatten(), linear]))

    # newmodel = torch.nn.Sequential(*(list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1]))))
    newmodel.cuda()
    newmodel = torch.nn.DataParallel(newmodel).cuda()


    wandb.watch(newmodel, log_freq=100)
    # print(newmodel)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(
        newmodel.parameters(), 
        5e-5
    )

    cudnn.benchmark = True

    dataset_folder = 'datasets/recorded_dataset_1641092589206/'
    dataset_file = dataset_folder + 'model.json'
    dataset = pd.read_json(dataset_file)

    epochs = 60
    step = 0


    
    for epoch in range(epochs):
        print("EPOCH ", epoch)
        newmodel.train()
        shapes = recorded_shapes(dataset, dataset_folder)
      
        for shape in shapes:
            (target, input) = shape
            output = newmodel(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step % 20 == 0):
                img = input.cpu().numpy()[0]
                img = np.transpose(img, (2, 1, 0)).copy() * 255
                data = [[label, val] for (label, val) in zip(shapeKeys, output.cpu().detach().numpy()[0].tolist())]
                table = wandb.Table(data=data, columns = ["label", "value"])
                data2 = [[label, val] for (label, val) in zip(shapeKeys, target.cpu().numpy()[0].tolist())]
                table2 = wandb.Table(data=data2, columns = ["label", "value"])
                wandb.log({
                    "image": wandb.Image(img), 
                    "actual_shape" : wandb.plot.bar(table2, "label", "value", title="Actual Shapes"),
                    "prediction_shapes" : wandb.plot.bar(table, "label", "value", title="Prediction Shapes")
                }, step=step)
           
            
            wandb.log({
                # "prediction": wandb.Image(img), 
                "train/loss": float(loss),
                "train/lr": float(optimizer.param_groups[0]['lr']),
            }, step=step)

            step = step + 1

        save_checkpoint({
            'epoch': epoch,
            'state_dict': newmodel.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, folder=checkpoint_folder, filename=f"checkpoint-{epoch}.pth.tar")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
    main()