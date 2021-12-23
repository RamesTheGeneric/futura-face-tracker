import os
import cv2
import numpy as np
import random

number = random.randrange(1, 650)
img_noise_name = os.path.join("datasets/prepocessed_dataset_2021-11-24_03-17-47-802457", 'images_noise', f'img-{number:02d}.jpg')
img_noise = cv2.imread(img_noise_name)
img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2GRAY)
img_noise = cv2.merge([img_noise,img_noise,img_noise]).astype(dtype=np.float32)
img = cv2.imread("datasets/prepocessed_dataset_2021-11-24_03-17-47-802457/images1024x1024/00000/00400.png", cv2.IMREAD_GRAYSCALE)
img = cv2.merge([img,img,img]).astype(dtype=np.float32)


image_norm = img * 2 / 255.0
noise_norm = img_noise * 3 / 255.0


img = np.clip((image_norm * (1 - (noise_norm))) * 255.0, 0, 255)
# for i in range(240):
#     dark = random.randrange(0, 255)
#     cv2.line(overlay, (i, 0), (i, 240), (dark, dark, dark), random.randrange(1, 4), cv2.LINE_AA)

# Transparency value
# alpha = random.randrange(2, 6) / 10

# Perform weighted addition of the input image and the overlay
# img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# img = cv2.convertScaleAbs(img, alpha=random.randrange(6, 10) / 10, beta=0)

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# hsv[:,:,2] += 50
# img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
img = cv2.resize(img, dsize=(240, 240))
cv2.imshow('frame', img.astype(dtype=np.uint8))
cv2.waitKey(0) 