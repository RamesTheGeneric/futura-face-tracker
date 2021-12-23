import pickle
import cv2
from simplejpeg import encode_jpeg
import pandas as pd

writer = open("image_archive.fia", "wb")
pointers = []
pointer = 0

dataset_folder = 'datasets/recorded_dataset_1640050443080/'
dataset_file = dataset_folder + 'model.json'
json = pd.read_json(dataset_file)

for image_name, data in json.iterrows():
    image_path = dataset_folder + image_name
    shape = data['blendshapes']['shape']
    image = cv2.imread(image_path)
    image_data = encode_jpeg(image)
    writer.write(image_data)
    pointers.append((pointer, len(image_data)))
    pointer += len(image_data)

# with open("image_archive.fia.metadata", "wb") as wb:
#     pickle.dump(pointers, wb)
# writer.close()