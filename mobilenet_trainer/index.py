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
from utils import save_checkpoint
from transformers import Rescale, ToTensor, Normalize

def get_dataloader(dataset):
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, sampler=train_sampler)
    return train_loader, len(train_loader)

def main(settings):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    name = 'Mobilenet-ffhq-croped-' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S-%f')
    checkpoint_folder = 'trained_models/Mobilenet-ffhq-croped/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    wandb.init(project="FuturaFaceTacker", name=name)

    model = mobilenetv2(num_classes=20 * 2)

    if not settings.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        settings.lr,
        # momentum=settings.momentum,
        weight_decay=settings.weight_decay)

    # if (settings.resume):
    #     checkpoint = torch.load('trained_models/Mobilenet-ffhq-croped/2021-11-20_17-25-01-155045/checkpoint.pth.tar')
    #     settings.start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
  
    cudnn.benchmark = True

    face_dataset = FaceLandmarksDataset(
        json_file='datasets/prepocessed_dataset_2021-12-16_22-48-29-975311/dataset.json', 
        root_dir='datasets/prepocessed_dataset_2021-12-16_22-48-29-975311', 
        transform=transforms.Compose([Rescale(224), Normalize(), ToTensor()])
    )
    train_loader, train_loader_len = get_dataloader(face_dataset)

    for epoch in range(settings.start_epoch, settings.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, settings.epochs))
        train_loss = train(settings, train_loader, train_loader_len, model, criterion, optimizer, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'mobilenetv2',
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, folder=checkpoint_folder, filename=f"checkpoint-{epoch}.pth.tar")

    return


if __name__ == '__main__':
    class Settings:
        world_size = -1
        lr = 0.00001
        momentum = 0.9
        distributed = False
        weight_decay = 1e-4
        resume = False
        start_epoch = 0
        epochs = 50
        warmup = False
        lr_decay = 'step'
        gamma = 0.1
    main(Settings())