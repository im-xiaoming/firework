import torch
import torch.nn as nn
from .function import softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, outputs, targets):
            assert outputs.ndim == 2, "Outputs must be 2-dimensions"
            assert targets.ndim == 1, "Targets must be 1-dimensions"
            
            logits = softmax(outputs) 
            filters = logits[torch.arange(targets.size(0)), targets]
            return torch.mean(-torch.log(filters))