import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import cv2
import random
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
        # number = random.randrange(1, 650)
        # img_noise_name = os.path.join(self.root_dir, 'images_noise', f'img-{number:02d}.jpg')
        # print(img_noise_name)
        # img_noise = cv2.imread(img_noise_name)
        # img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY)
        # img_noise = cv2.merge([img_noise,img_noise,img_noise]).astype(dtype=np.float32)
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.merge([img,img,img]).astype(dtype=np.float32)

        # Random brightness
        # image = cv2.convertScaleAbs(image, alpha=random.randrange(6, 10) / 10, beta=0)

        # overlay = image.copy()
        # for i in range(240):
        #     dark = random.randrange(30, 255)
        #     cv2.line(overlay, (i, 0), (i, 240), (dark, dark, dark), random.randrange(1, 3), cv2.LINE_AA)

        # # Transparency value

        # # Perform weighted addition of the input image and the overlay
        # image = cv2.addWeighted(img_noise, alpha, image, 1 - alpha, 0)

       
        # image_norm = image * 2 / 255.0
        # noise_norm = img_noise * 3 / 255.0


        # image = np.clip((image_norm * (1 - (noise_norm))) * 255.0, 0, 255)

        # image = np.clip(image + img_noise, 0, 255)


        landmarks = self.landmarks_frame.iloc[idx]['landmarks']
        landmarks = np.array(landmarks)[-20 * 2:]
        # landmarks = landmarks + np.random.randn(*landmarks.shape) * 0.4
        landmarks = landmarks.reshape((20, 2))
        # for index in range(68):
        #     if (index >= 0 and index <= 45):
        #         landmarks[index][0] = 0
        #         landmarks[index][1] = 0
                
        # emotions = np.array(self.landmarks_frame.iloc[idx]['emotions']).reshape(8)
        # print(emotions.shape)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample