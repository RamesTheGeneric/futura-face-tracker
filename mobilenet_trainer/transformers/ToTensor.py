import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample['image'] = torch.from_numpy(np.transpose(sample['image'], (2, 1, 0))).float()
        # sample['target'] = torch.from_numpy(np.concatenate((sample['landmarks'].reshape(136), sample['emotions']))).float()
        sample['target'] = torch.from_numpy(sample['landmarks']).reshape(20 * 2).float()
        sample['target'] = sample['target'] + torch.randn_like(sample['target']) * 0.01
        # print(sample['target'].shape)
        return sample
