import json
import os
import glob
import pandas as pd

dataset_folder = 'preprocessed_dataset'
files = glob.iglob(f"{dataset_folder}/**/*.json", recursive=True)

shapes = []
for file in files:
    with open(file) as json_file:
        data = json.load(json_file)
        shapes.append(data)

with open('preprocessed_dataset/dataset.json', 'w') as outfile:
    json.dump(shapes, outfile)