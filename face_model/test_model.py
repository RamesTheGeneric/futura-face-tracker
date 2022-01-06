from pyrender.mesh import Mesh
import trimesh
import pyrender
import numpy as np


facemodel = trimesh.load('face_model/facefull.glb')
# bottle_trimesh = facemodel.geometry[list(facemodel.geometry.keys())[0]]
mesh = Mesh.from_trimesh(facemodel.geometry['Wolf3D_Head'])
# mesh.weights[0] = 1
print(facemodel.geometry['Wolf3D_Head'])
# facemodel.geometry.weights[3] = 1
# mesh = Mesh.from_trimesh(list(facemodel.geometry.values()))
scene = pyrender.Scene()
scene.add(mesh)
v = pyrender.Viewer(scene, use_raymond_lighting=True)