import os
import errno
import torch

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_checkpoint(state, folder='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        mkdir_p(folder)
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
