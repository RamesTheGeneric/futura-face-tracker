import torch

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
