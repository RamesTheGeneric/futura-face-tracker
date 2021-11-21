from datetime import datetime
from multiprocessing import Pool
from numpy.core.fromnumeric import repeat
from functools import partial
import pandas as pd
import numpy as np
import cv2
import os
import dlib
import json

def findPointsBounds(points, offset):
    bot_left_x = min(point for point in points[0])
    bot_left_y = min(point for point in points[1])
    top_right_x = max(point for point in points[0])
    top_right_y = max(point for point in points[1])

    return [[int(bot_left_x - offset), int(bot_left_y - offset)], [int(top_right_x + offset), int(top_right_y + offset)]]


def process_data(data, context):
    try
        (face_detector, face_regressor, output_folder, dataset_folder) = context
        image = data['file_path']
        output_size = 240

        img = cv2.imread(os.path.join(dataset_folder, image))
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

        img_path = os.path.join(output_folder, image)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

        cv2.imwrite(img_path, graysacale) 
        # print('done', image)
        chunk = { "image": image, "landmarks": offseted_keypoints.tolist(), "original_points": pts.tolist() }
        with open(img_path + '.json', 'w') as outfile:
            json.dump(chunk, outfile)
            print(image)
    except Exception as e:
            print('error with image', image, e)

def main():
    dlib_landmark_model = 'trained_models/dlib/shape_predictor_68_face_landmarks.dat'
    
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    output_folder = 'datasets/prepocessed_dataset_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    dataset_folder = 'datasets/ffhq'
    

    # with open(os.path.join(dataset_folder, 'ffhq-dataset-v2.json')) as json_file:
        # ffhq_dataset = json.load(json_file)
    ffhq_dataset = pd.read_json(os.path.join(dataset_folder, 'ffhq-dataset-v2.json'), orient="index")
    print(ffhq_dataset['image'])
    with Pool(32) as p:
        p.map(partial(process_data, context=(face_detector, face_regressor, output_folder, dataset_folder)), ffhq_dataset['image'])


if __name__ == '__main__':
    main()