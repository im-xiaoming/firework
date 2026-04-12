import torch
import torch.nn as nn
from .function import softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        x = outputs - torch.max(outputs, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(x)
        probs = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

        filters = probs[torch.arange(targets.size(0)), targets]

        eps = 1e-9
        loss = -torch.log(filters + eps)

        return loss.mean()