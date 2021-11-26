import json
import os
import glob
import pandas as pd

dataset_folder = 'datasets/prepocessed_dataset_2021-11-24_16-35-11-386760'
files = glob.iglob(f"{dataset_folder}/images1024x1024/**/*.json", recursive=True)

shapes = []
for file in files:
    with open(file) as json_file:
        data = json.load(json_file)
        # print(data)
        # data['emotions'] = data['landmarks'][-8:]
        # data['landmarks'] = data['landmarks'][:-8]
        shapes.append(data)

with open(os.path.join(dataset_folder, 'dataset.json'), 'w') as outfile:
    json.dump(shapes, outfile)