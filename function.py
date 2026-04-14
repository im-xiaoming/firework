import torch
import torch.nn as nn

def log_softmax(x):
    log_probs = x - torch.max(x, dim=-1, keepdim=True)[0]
    log_probs = log_probs - torch.log(
        torch.sum(torch.exp(log_probs), dim=-1, keepdim=True)
    )
    return log_probs


class TaylorSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(TaylorSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x = x - x_max
        
        denorm = 1 + x + 0.5 * (x**2)
        
        probs = denorm / (denorm.sum(dim=self.dim, keepdim=True) + 1e-7)
        return probs


def log_soft_margin_softmax(x, y, m):
    x = x - torch.max(x, dim=-1, keepdim=True)[0]
    one_hot = torch.zeros_like(x)
    one_hot.scatter_(1, y.reshape(-1, 1), 1.0)

    margin = x - m * one_hot
    return margin - torch.log(torch.sum(torch.exp(margin), dim=-1, keepdim=True))
    
        