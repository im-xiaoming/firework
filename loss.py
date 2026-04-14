import torch
import torch.nn as nn
from .function import log_softmax, log_soft_margin_softmax, TaylorSoftmax
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        log_probs = log_softmax(outputs)

        loss = -log_probs[torch.arange(targets.size(0)), targets]
        return loss.mean()
    
    
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2):
        super(TaylorCrossEntropyLoss, self).__init__()
        self.softmax = TaylorSoftmax(dim=-1)

    def forward(self, outputs, targets):
        probs = self.softmax(outputs)
        
        log_probs = torch.log(probs + 1e-9)
        
        loss = F.nll_loss(log_probs, targets)
        return loss
    
    
class SMCrossEntropyLoss(nn.Module):
    def __init__(self, m=0.6):
        super().__init__()
        self.m = m

    def forward(self, outputs, targets):
        probs = log_soft_margin_softmax(outputs, targets, self.m)
        loss = -probs[torch.arange(outputs.size(0)), targets]
        return loss.mean()