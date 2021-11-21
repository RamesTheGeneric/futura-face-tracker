import torch
import numpy as np
from torchvision import transforms

convert_tensor = transforms.ToTensor()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.transpose(image, (2, 1, 0)) 
        tensor = torch.from_numpy(image)
        return {'image': tensor.float(), 'landmarks': torch.from_numpy(landmarks).reshape(136).float()}
