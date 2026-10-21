import torch
from torch import nn

class TinyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(3, 1)
        
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.2, 0.3, 0.5]])
            )
            self.linear.bias.fill_(0.1)
            
    def forward(self, features):
        return self.linear(features)