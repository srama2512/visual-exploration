import torch.nn as nn
from base.common import *

class BaseTaskHead(nn.Module):
    # The task prediction head
    def __init__(self, hidden_size, n_output):
        super(BaseTaskHead, self).__init__()
        self.n_output = n_output
        final_layer = ixvr(nn.Linear(hidden_size, self.n_output))
        self.predict = nn.Sequential(nn.BatchNorm1d(hidden_size), 
                                     final_layer)
    def forward(self, x):
        return self.predict(x)
