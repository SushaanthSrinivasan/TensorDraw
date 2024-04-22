import torch
import torch.nn as nn
import sys
sys.path.append("./")
from tensordraw import *

from create_test_model import VGG16

model = VGG16()

draw_network(model, 'test_model_vgg.pth', 'torch')