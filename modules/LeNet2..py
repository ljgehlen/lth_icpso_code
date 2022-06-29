import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        model = nn.Sequential(nn.Conv2d(1, 6, 5),
                      nn.ReLU(),
                      nn.MaxPool2d((2,2)),
                      nn.Conv2d(6, 16, 5),
                      nn.ReLU(),
                      nn.MaxPool2d(2),
                      nn.Flatten(1),
                      nn.Linear(16 * 5 * 5, 120),
                      nn.ReLU(),
                      nn.Linear(120, 84),
                      nn.ReLU(),
                      nn.Linear(84, 10)
                     )