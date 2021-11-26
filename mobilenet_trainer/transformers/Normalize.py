import torch

class Normalize(object):
    """Normalize landmarks"""

    def __call__(self, sample):
        width = sample['image'].shape[0]
        height = sample['image'].shape[1]
        sample['image'] = sample['image'] / 255
        sample['landmarks'] = sample['landmarks'] / (width, height) 
        return sample
