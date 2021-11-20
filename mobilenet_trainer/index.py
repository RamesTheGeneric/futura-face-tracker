import torch
import os
from torch import nn
from torch.backends import cudnn
from torch.utils.data.dataloader import DataLoader
import wandb
from datetime import datetime
import torchvision.transforms as transforms
from FaceLandmarksDataset import FaceLandmarksDataset
from mobilenetV2 import mobilenetv2
from train import train
from utils import PrefetchedWrapper, save_checkpoint
from transformers import Rescale, ToTensor



def get_dataloader(dataset):
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, sampler=train_sampler)
    return PrefetchedWrapper(train_loader), len(train_loader)
    

def main(settings):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    name = 'Mobilenet-ffhq-croped-' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S-%f')
    checkpoint_folder = 'trained_models/Mobilenet-ffhq-croped/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    wandb.init(project="FuturaFaceTacker", name=name)

    model = mobilenetv2()

    if not settings.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        settings.lr,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay)
    cudnn.benchmark = True

    face_dataset = FaceLandmarksDataset(
        json_file='datasets/preprocessed_dataset/dataset.json', 
        root_dir='datasets/preprocessed_dataset', 
        transform=transforms.Compose([Rescale(224), ToTensor()])
    )
    train_loader, train_loader_len = get_dataloader(face_dataset)

    best_prec1 = 0
    for epoch in range(settings.start_epoch, settings.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, settings.epochs))
        train_loss = train(settings, train_loader, train_loader_len, model, criterion, optimizer, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'mobilenetv2',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, folder=checkpoint_folder)

    return


if __name__ == '__main__':
    class Settings:
        world_size = -1
        lr = 0.1
        momentum = 0.9
        distributed = False
        weight_decay = 1e-4
        start_epoch = 0
        epochs = 90
        warmup = False
        lr_decay = 'step'
        gamma = 0.1
    main(Settings())