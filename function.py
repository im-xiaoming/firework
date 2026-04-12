import torch

def softmax(x):
    logits = torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True) 
    return logits