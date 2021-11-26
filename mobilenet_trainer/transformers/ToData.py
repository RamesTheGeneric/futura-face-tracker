

import numpy as np


def tensorToData(input, output, target):
    img = input.cpu().numpy()[0]
    img = np.transpose(img, (2, 1, 0)).copy() * 255
    target = target.cpu().numpy()[0]
    output = output.cpu().detach().numpy()[0]
    # outputLandmarks = output[:-8].reshape((68, 2))
    # targetLandmarks = target[:-8].reshape((68, 2))
    # outputEmotions = output[-8:]
    # targetEmotions = target[-8:]
    outputLandmarks = output.reshape((20, 2))
    targetLandmarks = target.reshape((20, 2))
    # outputEmotions = output[-8:]
    # targetEmotions = target[-8:]
    return (img, outputLandmarks, targetLandmarks)