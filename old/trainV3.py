from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import torch.nn as nn
import time
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from math import cos, pi
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import wandb

import models.imagenet as customized_models

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

plt.ion()   # interactive mode


default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e

class PrefetchedWrapper(object):
    def prefetched_loader(loader):
        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)

        stream = torch.cuda.Stream()
        first = True

        for entry in loader:
            with torch.cuda.stream(stream):
                next_input = entry['image'].cuda(non_blocking=True)
                next_target = entry['landmarks'].cuda(non_blocking=True)
                next_input = next_input.float()
                # if one_hot:
                    # next_target = expand(num_classes, torch.float, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.epoch = 0

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):

            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader
        )

    def __len__(self):
        return len(self.dataloader)

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx]['image'])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx]['landmarks']
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks / (new_w, new_h)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # print(image.shape) # (240, 240)
        tensor = torch.from_numpy(image)
        tensor = torch.stack([tensor, tensor, tensor], dim=0)
        return {'image': tensor.float(), 'landmarks': torch.from_numpy(landmarks).reshape((136)).float()}


def get_dataloaders(dataset):
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
   
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, sampler=train_sampler)
    return PrefetchedWrapper(train_loader), len(train_loader), PrefetchedWrapper(test_loader), len(test_loader)
    


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 1



    grid = utils.make_grid(images_batch, nrow=batch_size)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def adjust_learning_rate(settings, optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if settings.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = settings.epochs * num_iter

    if settings.lr_decay == 'step':
        lr = settings.lr * (settings.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif settings.lr_decay == 'cos':
        lr = settings.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif settings.lr_decay == 'linear':
        lr = settings.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif settings.lr_decay == 'schedule':
        count = sum([1 for s in settings.schedule if s <= epoch])
        lr = settings.lr * pow(settings.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(settings.lr_decay))

    if epoch < warmup_epoch:
        lr = settings.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(settings, train_loader, train_loader_len, model, criterion, optimizer, epoch):
    bar = Bar('Processing', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(settings, optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        wandb.log({ "train/loss": float(loss) })
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=i + 1,
            size=train_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def validate(val_loader, val_loader_len, model, criterion):
    bar = Bar('Validating', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1.item(), input.size(0))
        # top5.update(prec5.item(), input.size(0))

        wandb.log({ "validation/loss": float(loss) })
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=i + 1,
            size=val_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



def main(settings):
    best_prec1 = 0
    print(settings)
    # create model
    print("=> creating model '{}'".format(settings.arch))
    model = models.__dict__[settings.arch]()

    settings.distributed = settings.world_size > 1
    print(torch.cuda.is_available())
    if not settings.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    title = 'ImageNet-' + settings.arch
    if not os.path.isdir(settings.checkpoint):
        mkdir_p(settings.checkpoint)


    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), 
        settings.lr,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay)

    cudnn.benchmark = True

    face_dataset = FaceLandmarksDataset(
        json_file='preprocessed_dataset/dataset.json', 
        root_dir='preprocessed_dataset', 
        transform=transforms.Compose([Rescale(224), ToTensor()])
    )
    train_loader, train_loader_len, val_loader, val_loader_len = get_dataloaders(face_dataset)
    # val_loader, val_loader_len = get_train_dataloader(face_dataset)
    wandb.init(project="FuturaFaceTacker", name=title)

    for epoch in range(settings.start_epoch, settings.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, settings.epochs))

        # train for one epoch
        train_loss, train_acc = train(settings, train_loader, train_loader_len, model, criterion, optimizer, epoch)
        
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)

        lr = optimizer.param_groups[0]['lr']
        # logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        # writer.add_scalar('learning rate', lr, epoch + 1)
        # writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        # writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': settings.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=settings.checkpoint)
    # logger.close()
    # logger.plot()
    savefig(os.path.join(settings.checkpoint, 'log.eps'))
    # writer.close()

    print('Best accuracy:')
    print(best_prec1)

if __name__ == '__main__':
    class Settings:
        arch = "mobilenetv2"
        world_size = -1
        lr = 0.1
        momentum = 0.9
        weight_decay = 1e-4
        checkpoint = 'checkpoints'
        start_epoch = 0
        epochs = 90
        warmup = False
        lr_decay = 'step'
        gamma = 0.1
        schedule = [150, 225]


    main(Settings())


    