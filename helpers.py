import subprocess
import os
import glob

def find_group_indices(layer_list, lib):
    group_indices, conv_layers = [], []
    start_index = None
    end_index = None

    if lib == 'tf':
        pass
    elif lib == 'torch':
        import torch
        import torch.nn as nn

        # Iterate through the layers to find the conv layers up to the first max pool layer
        for idx, (name, layer) in enumerate(layer_list):
            if isinstance(layer, nn.Conv2d):
                # If the start index is None, set it to the current index
                if start_index is None:
                    start_index = idx
                # Append the conv layer to the conv_layers list
                conv_layers.append(layer)
            elif isinstance(layer, nn.MaxPool2d):
                # If there are conv layers before the max pool layer
                if start_index is not None:
                    # Set the end index to the current index
                    end_index = idx
                    # Process the group of conv layers
                    print(f"Group of Conv layers from index {start_index} to {end_index}")
                    group_indices.append((start_index, end_index+1))
                    for conv_layer in conv_layers:
                        print(conv_layer)
                    # Reset the start and end indices for the next group
                    start_index = None
                    end_index = None
                    conv_layers = []
    return group_indices

def consecutive_count(arr):
    consecutive_groups = []
    start_index = None
    count = 0

    for i, num in enumerate(arr):
        if start_index is None:
            start_index = i
            count = 1
        elif num == arr[i - 1] + 1:
            count += 1
        else:
            consecutive_groups.append((start_index, count))
            start_index = i
            count = 1

    if start_index is not None:
        consecutive_groups.append((start_index, count))

    return consecutive_groups

def get_convrelu_subgroups(layer_list):
    indices = []
    for index, layer in enumerate(layer_list[:-1]):  # Ignore the last layer
        if isinstance(layer, nn.Conv2d) and isinstance(layer_list[index + 1], nn.ReLU):
            indices.append(index)
    counts = consecutive_count(indices)
    return counts

def open_pdf(file_path):
    try:
        # Open the PDF file in the default PDF viewer
        subprocess.run(['start', '', file_path], check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unable to open PDF file: {e}")

def run_pdflatex(file_name):
    try:
        subprocess.run(['pdflatex', file_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Unable to run pdflatex: {e}")

def cleanup_files(path):
    # patterns = ['*.aux', '*.log', '*.vscodeLog', '*.tex']
    patterns = ['*.aux', '*.log', '*.vscodeLog']
    for pattern in patterns:
        files_to_delete = glob.glob(os.path.join(path, pattern))
        for file in files_to_delete:
            os.remove(file)
