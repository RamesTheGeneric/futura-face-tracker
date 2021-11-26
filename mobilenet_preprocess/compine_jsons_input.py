import json
import os
import glob
import pandas as pd
import numpy as np

dataset_folder = 'datasets/ffhq/json'
files = glob.iglob(f"{dataset_folder}/**/*.json", recursive=True)

shapes = []
i = 0
for file in files:
    with open(file) as json_file:
        data = json.load(json_file)
        if (len(data) > 0):
            d = data[0]
            d['image'] = int(os.path.basename(file)[:4])
            shapes.append(d)
        # else:
            # shapes.append({})

print(len(shapes))
with open('datasets/ffhq/ffhq-extension.json', 'w') as outfile:
    json.dump(shapes, outfile)