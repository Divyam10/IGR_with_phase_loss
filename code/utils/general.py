import os

import numpy as np
import torch
import trimesh
import pandas as pd 

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def load_point_cloud_by_file_extension(file_name, with_normals):

    
    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    
    # elif self.with_normals:
    #Trimesh does not load normals from xyz files. Loading it using pandas.
    elif with_normals:

        point_set = pd.read_table("/home/ai/Desktop/Preimage_Implicit_DLTaskData/bunny_normals_50000.xyz", skiprows=0, delim_whitespace=True,
                         names=['x', 'y', 'z', 'nx', 'ny', 'nz'])

        return torch.tensor(point_set.values).float()
    
    else:
        point_set = torch.tensor(trimesh.load(file_name, ext).vertices).float()
        
#         min_x, max_x = np.min(point_set[:, 0], axis=0), np.max(point_set[:, 0], axis=0)
#         min_y, max_y = np.min(point_set[:, 1], axis=0), np.max(point_set[:, 1], axis=0)
#         min_z, max_z = np.min(point_set[:, 2], axis=0), np.max(point_set[:, 2], axis=0)
    
#         normalized_x = 1 * (point_set[:, 0] - min_x) / (max_x - min_x) - 1
#         normalized_y = 1 * (point_set[:, 1] - min_y) / (max_x - min_x) - 1
#         normalized_z = 1 * (point_set[:, 2] - min_z) / (max_x - min_x) - 1
        
#         point_set = np.column_stack((normalized_x, normalized_y, normalized_z))
            
#         return torch.tensor(point_set).float()
        
        return point_set

class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)
