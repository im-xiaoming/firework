import torch
import torch.nn as nn
from .function import log_softmax, taylor_softmax, soft_margin_softmax

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, outputs, targets):
        log_probs = log_softmax(outputs)

        loss = -log_probs[torch.arange(targets.size(0)), targets]
        return loss.mean()
    
    
class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(TaylorCrossEntropyLoss, self).__init__()
        
    def forward(self, outputs, targets):
        probs = taylor_softmax(outputs)
        loss = -probs[torch.arange(outputs.size(0)), targets]
        
        return loss.mean()
    
    
class SMCrossEntropyLoss(nn.Module):
    def __init__(self, m=0.6):
        super(TaylorCrossEntropyLoss, self).__init__()
        self.m = m
        
    def forward(self, outputs, targets):
        probs = soft_margin_softmax(outputs, targets, self.m)
        loss = -probs[torch.arange(outputs.size(0)), targets]
        
        return loss.mean()