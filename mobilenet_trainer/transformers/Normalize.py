import torch

class Normalize(object):
    """Normalize landmarks"""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        width = image.shape[0]
        height = image.shape[1]
        return {'image': image / 255, 'landmarks': landmarks / (width, height) }
