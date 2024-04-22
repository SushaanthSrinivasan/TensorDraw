import sys
sys.path.append("./")
from PlotNeuralNet.pycore.tikzeng import *
from helpers import *

def draw_group(layer_list, group_indices):
    # TODO
    arch = []
    (start_idx, end_idx) = group_indices

    offset_amount = 0

    counts = get_convrelu_subgroups(layer_list)
    for idx, count in counts:
        arch.append(to_ConvNRelu(name, layer.out_channels, layer.kernel_size, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=64, width=2))

    i = start_idx
    while i <= end_idx:
        name, layer = layer_list[i][0], layer_list[i][1]
        if i < len(layer_list) - 1:
            next_name, next_layer = layer_list[i+1]
        else:
            next_name, next_layer = None, None
        if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.ReLU):
            arch.append(to_ConvRelu(name, layer.out_channels, layer.kernel_size, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=64, width=2))
        elif isinstance(layer, nn.Conv2d):
            arch.append(to_Conv(name, layer.out_channels, layer.kernel_size, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=64, width=2))
        elif isinstance(layer, nn.MaxPool2d):
            arch.append(to_Pool(name, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=60, depth=10, opacity=0.8))
        elif isinstance(layer, nn.Linear):
            arch.append(to_Linear(name, 256, 64, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=10, width=2))
        i += 1
        offset_amount += 1
    return arch

def draw_network(model, weights_path, lib):
    try:
        arch = [
            to_head('./PlotNeuralNet'),
            to_cor(),
            to_begin()
        ]

        if lib == 'tf':
            pass
        elif lib == 'torch':
            import torch
            import torch.nn as nn

            offset_amount = 1
            increment = 1
            multiplier = 1

            model.load_state_dict(torch.load(weights_path))

            children_list = []
            for index, (name, layer) in enumerate(model.named_children()):
                if isinstance(layer, nn.Sequential):
                    for subindex, (sublayername, sublayer) in enumerate(layer.named_children()):
                        children_list.append((sublayername, sublayer))
                else:
                    children_list.append((name, layer))

            # print(*children_list, sep="\n") 
            # exit()

            group_indices = find_group_indices(children_list, 'torch')
            gi_pos = 0
            print(group_indices)
            print(len(children_list))

            for index, (name, layer) in enumerate(children_list):
                print(name, layer)
                if index == group_indices[gi_pos][0]: # group start
                    group_arch = draw_group(children_list, group_indices[gi_pos])
                    print("group_arch")
                    print(*group_arch, sep="\n")
                    # exit()
                    arch = arch + group_arch
                    break
                    # exit()
                elif index == group_indices[gi_pos][1]: # group end
                    # TODO
                    continue
                elif index in range(group_indices[gi_pos][0], group_indices[gi_pos][0]): # group middle
                    # TODO
                    continue
                else: # not group
                    # TODO
                    print("NOT GROUP")
                    if index < len(list(model.named_children())) - 1:
                        next_name, next_layer = list(model.named_children())[index + 1]
                    else:
                        next_name, next_layer = None, None

                    if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.ReLU):
                        arch.append(to_ConvRelu(name, layer.out_channels, layer.kernel_size, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=64, width=2))
                    
                    elif isinstance(layer, nn.Conv2d):
                        arch.append(to_Conv(name, layer.out_channels, layer.kernel_size, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=64, width=2))
                    
                    elif isinstance(layer, nn.Linear):
                        arch.append(to_Linear(name, 256, 64, offset=f"(0,0,0)", to=f"({offset_amount},0,0)", height=64, depth=10, width=2))
                        multiplier = 2
                    
                    print(f'Position of latest layer: ({offset_amount},0,0)')
                    
                    if (index+1) % 2 == 0:
                        print(f'Drawing connection between ({offset_amount-(increment*multiplier)},0,0) and ({offset_amount+(increment*multiplier)},0,0)')
                        arch.append(to_connection(f"({offset_amount-(increment*multiplier)},0,0)", f"({offset_amount+(increment*multiplier)},0,0)"))
                    
                    offset_amount += increment * multiplier
                    multiplier = 1

        arch.append(to_end())

        filename = "testarch"
        to_generate(arch, f'{filename}.tex')

        run_pdflatex(f'{filename}.tex')

        open_pdf(f"{filename}.pdf")
    finally:
        cleanup_files('.')