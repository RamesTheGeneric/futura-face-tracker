import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import cv2
from PIL import Image

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
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.merge([img,img,img])
        landmarks = self.landmarks_frame.iloc[idx]['landmarks']
        landmarks = np.array(landmarks)
        landmarks = landmarks.reshape((68, 2))
        for index in range(68):
            if (index >= 0 and index <= 35):
                landmarks[index][0] = 0
                landmarks[index][1] = 0
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample