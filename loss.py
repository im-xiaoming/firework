import torch
import torch.nn as nn
from .function import log_softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        log_probs = log_softmax(outputs)

        loss = -log_probs[torch.arange(targets.size(0)), targets]
        return loss.mean()