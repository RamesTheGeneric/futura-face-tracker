from threading import Thread
import cv2
import numpy as np
import cv2, os

import tensorflow

class config:
    X_RES = 128
    Y_RES = 128

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model = tensorflow.keras.models.load_model('saved_model')

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        img = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, dsize=(config.X_RES, config.Y_RES))
        data_img = np.reshape(img, (config.X_RES, config.Y_RES, 1))
        images = np.array([data_img])/255.
        points = model.predict(images)[0]
        for i in range(int(len(points) / 2)):
            cv2.circle(img, (int(points[i * 2]) , int(points[i * 2 + 1])) , 1, (255, 0, 0), 2)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    src = 'http://192.168.1.17/stream'
    threaded_camera = ThreadedCamera(src)
    while True:
        try:
            threaded_camera.show_frame()
        except AttributeError:
            pass