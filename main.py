import torch
import torch.nn as nn
import sys
import os
import glob

sys.path.append("./")

from PlotNeuralNet.pycore.tikzeng import *
from helpers import *

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the saved model
model = SimpleCNN()
model.load_state_dict(torch.load('test_model.pth'))

arch = [
    to_head( './PlotNeuralNet' ),
    to_cor(),
    to_begin()
]

# Extract information about each layer
print("Layer Information:")
for name, layer in model.named_children():


    if isinstance(layer, nn.Conv2d):
        arch.append(to_Conv(name, layer.out_channels, layer.kernel_size, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2))

        
arch.append(to_end())
filename = "testarch"
to_generate(arch, f'{filename}.tex')

run_pdflatex(f'{filename}.tex')

open_pdf(f"{filename}.pdf")
cleanup_files('.')