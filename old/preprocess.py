import json
import numpy as np
import cv2
import dlib
import pandas as pd
import os

import concurrent.futures

def findPointsBounds(points, offset):
    bot_left_x = min(point for point in points[0])
    bot_left_y = min(point for point in points[1])
    top_right_x = max(point for point in points[0])
    top_right_y = max(point for point in points[1])

    return [[int(bot_left_x - offset), int(bot_left_y - offset)], [int(top_right_x + offset), int(top_right_y + offset)]]


dlib_landmark_model = '3DDFA/models/shape_predictor_68_face_landmarks.dat'
face_regressor = dlib.shape_predictor(dlib_landmark_model)
face_detector = dlib.get_frontal_face_detector()

ffhq_model = 'ffhq-dataset-v2.json'

train_data = pd.read_json(ffhq_model, orient='index')


def large_list_generator_func(data):

    image = data['image']['file_path']
    output_size = 240

    img = cv2.imread(image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = face_detector(RGB_img, 1)
    if len(rects) == 0:
        print("skip image %s, could not find landmarks" % (image))
        return False

    rects = sorted(rects, key=lambda x: x.width() + x.height(), reverse=True)

    pts = face_regressor(RGB_img, rects[0]).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T

    [a, b] = findPointsBounds([pts[0][48::], pts[1][48::]], 80)

    middleX = (b[0] - a[0]) / 2 + a[0]
    middleY = (b[1] - a[1]) / 2 + a[1]

    resHalf = max(output_size / 2, max(b[1] - a[1], b[0] - a[0])) / 2

    cornerA = [
        middleX - resHalf,
        middleY - resHalf
    ]

    cornerB = [
        middleX + resHalf,
        middleY + resHalf
    ]

    imgCropped = RGB_img[int(cornerA[1]):int(cornerB[1]), int(cornerA[0]):int(cornerB[0])]

    offseted_keypoints = []
    scale = output_size / (resHalf * 2)
    for index in range(0, 68):
        point = [pts[0][index] - cornerA[0], pts[1][index] - cornerA[1]]
        if point[0] < 0 or point[0] >= resHalf * 2 or point[1] < 0 or point[1] >= resHalf * 2:
            point = [0, 0]
        point = np.array(point) * scale
        offseted_keypoints.append(point)
    offseted_keypoints = np.array(offseted_keypoints).flatten()

    graysacale = cv2.cvtColor(imgCropped, cv2.COLOR_RGB2GRAY)
    graysacale = cv2.resize(graysacale, dsize=(output_size, output_size))

    image_save = 'preprocessed_dataset/' + image
    os.makedirs(os.path.dirname(image_save), exist_ok=True)

    cv2.imwrite(image_save, graysacale) 
    # print('done', image)
    chunk = { "image": image, "landmarks": offseted_keypoints.tolist() }
    with open('preprocessed_dataset/' + image + '.json', 'w') as outfile:
        json.dump(chunk, outfile)

def jsonIterator():
    for future in concurrent.futures.as_completed(futures):
        yield future.result()



if __name__ == "__main__":
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
    futures = []
    for i, d in train_data.iterrows():
        if os.path.isfile('preprocessed_dataset/' + d['image']['file_path'] + '.json'):
            print('skip existing', d['image']['file_path'])
            continue
        futures.append(executor.submit(large_list_generator_func, data=d))
    for future in concurrent.futures.as_completed(futures):
        res = future.result()
