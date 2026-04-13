import torch

def log_softmax(x):
    log_probs = x - torch.max(x, dim=-1, keepdim=True)[0]
    log_probs = log_probs - torch.log(
        torch.sum(torch.exp(log_probs), dim=-1, keepdim=True)
    )
    return log_probs


def taylor_softmax(x):
    probs = x - torch.max(x, dim=-1, keepdim=True)[0]
    probs = (1 + probs + 0.5 * probs**2).clamp(min=0)
    probs = (1 + probs + 0.5 * probs**2) / (torch.sum(1 + probs + 0.5 * probs**2, dim=-1, keepdim=True))
    return torch.log(probs)


def soft_margin_softmax(x, y, m):
    
    one_hot = torch.zeros_like(x, device=x.device)
    one_hot.scatter_(1, y.reshape(-1, 1), 1.0)
    
    margin = x * one_hot - m + x * (1 - one_hot)
    return margin - torch.log(
        torch.exp(margin) + torch.sum(margin, dim=-1, keepdim=True)
    )
    
        