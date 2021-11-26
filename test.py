import cv2
import numpy as np
import random



img = cv2.imread("datasets\\prepocessed_dataset_2021-11-24_16-35-11-386760\\images1024x1024\\00000\\00400.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.merge([img,img,img])
overlay = img.copy()



for i in range(15):
    y = random.randrange(0, 240)
    dark = random.randrange(0, 80)
    cv2.line(overlay, (0, y), (240, y), (dark, dark, dark), random.randrange(3, 8), cv2.LINE_AA)

# Transparency value
alpha = random.randrange(1, 3) / 10

# Perform weighted addition of the input image and the overlay
img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# hsv[:,:,2] += 50
# img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
img = cv2.resize(img, dsize=(240, 240))
cv2.imshow('frame', img)
cv2.waitKey(0) 