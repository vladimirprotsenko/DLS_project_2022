import torch
import torch.nn as nn


class MSELossRedefined(nn.Module):
    def __init__(self):
        super(MSELossRedefined, self).__init__()

    def __call__(self, input, is_true):
        if is_true:
            ground_true = torch.ones_like(input)
        else:
            ground_true = torch.zeros_like(input)
        loss = nn.MSELoss( )
        return loss(input, ground_true)