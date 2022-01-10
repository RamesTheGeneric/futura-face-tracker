
import enum
import torch
from torch.backends import cudnn
import os
from threading import Thread
import cv2
import numpy as np
import io
from torch.utils.data.dataset import Dataset
from mobilenetV2 import mobilenetv2
from utils import save_checkpoint
from torch import nn
import pandas as pd
from torch.utils.data.dataloader import DataLoader
import imageio
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
import random



class config:
    X_RES = 224
    Y_RES = 224
    POINTS = 20
    BLENDSHAPES = 37

def enable_gradient_layer(layer):
    layer.enable_gradient=False
    return layer

def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def recorded_shapes(dataset, dataset_folder, batch_size=64, agmentations=False):
    for record_name, b in dataset.iterrows():
        record_path = dataset_folder + record_name + '.mp4'
        shape = b['blendshapes']['shape']
        reader = imageio.get_reader(record_path)
        inputs = []
        targets = []

        image_count = 0

        for transform in range(2 if agmentations else 1):
            rand_rot = random.randrange(-25, 25)

            for frame_number, im in enumerate(reader):
                image = grayConversion(im)
                image = cv2.merge([image,image,image]).astype(dtype=np.float32)
                if (transform == 1):
                    image = rotate(image, rand_rot)
                image = cv2.resize(image, dsize=(config.X_RES, config.Y_RES))
                
                image = image / 255
                targets.append(torch.from_numpy(np.array(shape).astype(dtype=np.float32)).float())
                inputs.append(torch.from_numpy(np.transpose(image, (2, 1, 0))).float())
                image_count = image_count + 1

                if (image_count % batch_size == 0):
                    stacked = torch.stack(inputs, dim=0).cuda()
                    tstacked = torch.stack(targets, dim=0).cuda()
                    inputs = []
                    targets = []
                    image_count = 0
                    yield tstacked, stacked

def log_prediction(step, input, output, target, validation=False):
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

    img = input.cpu().numpy()[0]
    img = np.transpose(img, (2, 1, 0)).copy() * 255
    # if validation:
        # print(np.amin(img), np.amax(img), img)
    fig = Figure(figsize=(10, 10), dpi=60)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0,0,1,1])

    labels = shapeKeys
    out_l =  output.cpu().detach().numpy()[0]
    target_l = target.cpu().numpy()[0]
    # if validation:
    print("input", np.amin(input.cpu().numpy()[0]), np.amax(input.cpu().numpy()[0]), input.cpu().numpy()[0].shape)
    print("out", np.amin(out_l), np.amax(out_l), out_l.shape)
    print("target", np.amin(target_l), np.amax(target_l), target_l.shape)

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

    return img, pred



def train(model, step, shapes, criterion, optimizer):
    model.train()
    for shape in shapes:
        optimizer.zero_grad()
        (target, input) = shape
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if (step % 64 == 0):
            (img, pred) = log_prediction(step, input, output, target)

            wandb.log({
                f"prediction": [wandb.Image(img), wandb.Image(pred)], 
            }, step=step)
        
        wandb.log({
            # "prediction": wandb.Image(img), 
            "train/loss": float(loss),
            "train/lr": float(optimizer.param_groups[0]['lr']),
        }, step=step)

        step = step + 1

    return step

def validate(model, epoch, step, shapes):
    model.eval()
    list = []
    for shape in shapes:
        if (step % 64 == 0 and step < 500):
            (target, input) = shape
            output = model(input)
            (img, pred) = log_prediction(step, input, output, target, True)
            list = list +  [wandb.Image(img), wandb.Image(pred)]
        step = step + 1
       
    wandb.log({
        f"val_{epoch}": list, 
    })
    return step

def main():
    wandb.init(project="FuturaFaceTacker", name="FuturaFaceBlenshapes-" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S-%f'))

    checkpoint_folder = 'trained_models/FuturaFaceBlenshapes/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    model = mobilenetv2(num_classes=config.POINTS*2)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load('trained_models\\Mobilenet-ffhq-croped\\2021-12-16_23-48-32-851676\\checkpoint-40.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    linear = torch.nn.Linear(1280, config.BLENDSHAPES)
    newmodel = torch.nn.Sequential(*([torch.nn.ReLU()]  + list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1])) + [torch.nn.Flatten(), torch.nn.ReLU(), linear]))
    # # newmodel = torch.nn.Sequential(*(list(map(enable_gradient_layer, list(list(model.children())[0].children())[:-1]))))
    newmodel = torch.nn.DataParallel(newmodel).cuda()
    newmodel.cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(
        newmodel.parameters(), 
        5e-5,
    )

    resume = False
    start_epoch = 0
    if resume:
        newmodel_checkpoint = torch.load('trained_models\\FuturaFaceBlenshapes\\2022-01-09_09-12-05-386685\\checkpoint-27.pth.tar')
        newmodel.load_state_dict(newmodel_checkpoint['state_dict'])
        start_epoch = newmodel_checkpoint['epoch'] + 1
        optimizer.load_state_dict(newmodel_checkpoint['optimizer'])
        print("RESUMING FROM EPOCH", start_epoch)

    newmodel.eval()


    wandb.watch(newmodel, log_freq=100)
    cudnn.benchmark = True

    dataset_folder = 'datasets/recorded_dataset_1641092589206/'
    dataset_file = dataset_folder + 'model.json'
    dataset = pd.read_json(dataset_file)

    epochs = 60
    step = 0
   
    # dataset = dataset.sample(frac=0.2)
    train_samples = dataset.copy()
    validation_samples = dataset.copy()

    for epoch in range(epochs):
        if (epoch < start_epoch):
            continue
        print("EPOCH ", epoch)
        newmodel.train()
        train_shapes = recorded_shapes(train_samples, dataset_folder, 64)
        step = train(newmodel, step, train_shapes, criterion, optimizer)

        with torch.no_grad():
            newmodel.eval()
            print("VALIDATION ", epoch)
            val_shapes = recorded_shapes(validation_samples, dataset_folder, 64)
            val_step = 0
            val_step = validate(newmodel, epoch, val_step, val_shapes)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': newmodel.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, folder=checkpoint_folder, filename=f"checkpoint-{epoch}.pth.tar")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"  
    main()