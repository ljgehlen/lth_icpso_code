import icpso_values
import torch
from torch import nn

class Distribution:
    def __init__(self, chance_0, chance_1):
        self.chance_0 = chance_0
        self.chance_1 = chance_1

def count_layers(model):
	table = []
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		if "bias" in name: continue
		param = parameter.numel()
		table.append(param)
		total_params+=param
	print(table)
	return table, total_params

def main(model):


if __name__ == "__main__":
    main(model)
