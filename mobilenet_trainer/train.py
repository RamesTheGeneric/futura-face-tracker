import time
import PIL
import cv2

import torch
from transformers import tensorToData
from utils import AverageMeter
import numpy as np
import wandb
from progress.bar import Bar


def adjust_learning_rate(settings, optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if settings.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = settings.epochs * num_iter

    lr = settings.lr * (settings.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))

    if epoch < warmup_epoch:
        lr = settings.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
    count = 0
    for i, batch in enumerate(train_loader):
        input = batch['image']
        target = batch['target']
        step = count + (epoch * train_loader_len)

        lr = adjust_learning_rate(settings, optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
      
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (step % 100 == 0):
            (img, outputLandmarks, targetLandmarks) = tensorToData(input, output, target)
            for e in range(20):
                cv2.circle(img, (int(outputLandmarks[e][0] * 224) , int(outputLandmarks[e][1] * 224)) , 1, (255, 0, 0), 2)
                cv2.circle(img, (int(targetLandmarks[e][0] * 224) , int(targetLandmarks[e][1] * 224)) , 1, (0, 255, 0), 2)
            # tablePrediction = wandb.Table(data=[[label, val] for (label, val) in zip(, outputEmotions)], columns = ["emotion", "value"])
            # tableEmotions = wandb.Table(data=[[label, val] for (label, val) in zip(['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'], targetEmotions)], columns = ["emotion", "value"])
            # emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
            # wandb.log({'emotion/' + emotions[i] + '/target': targetEmotions[i] for i in range(len(emotions))}, step=step)
            # wandb.log({'emotion/' + emotions[i] + '/prediction': outputEmotions[i] for i in range(len(emotions))}, step=step)
            wandb.log({ 
                "prediction": wandb.Image(img), 
                # "emotions_prediction":  wandb.plot.bar(tablePrediction, "emotion", "value", title="Emotions Predictions"),
                # "emotions": wandb.plot.bar(tableEmotions, "emotion", "value", title="Emotions Target") 
            }, step=step)
        
        
        wandb.log({
            "train/loss": float(loss),
            "train/lr": float(lr),
        }, step=step)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | LR: {lr:.4f}'.format(
            batch=i + 1,
            size=train_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            lr=lr
        )
        count = count + 1
        bar.next()
    bar.finish()
    return (losses.avg)