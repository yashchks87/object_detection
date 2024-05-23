import torch
import vision
import torch.nn as nn
import torch.nn.functional as F

class BackBone:
    def __init__(self):
        # Required properties
        self.feature_map_channels = 0
        self.feature_pixels = 0
        self.feature_vector_size = 0
        # self.image_preprocessing_parameters = 

        # Required members
        self.feature_extractor = None
        self.pool_to_feature_vector = None
    
    def compute_feature_map_shape(self, image_shape):
        pass