import sys
sys.path.append("./")
from PlotNeuralNet.pycore.tikzeng import *
from helpers import *

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

            children_list = list(model.named_children())

            for index, (name, layer) in enumerate(model.named_children()):
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