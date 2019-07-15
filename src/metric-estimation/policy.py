import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.autograd import Variable
from base.base_policy import BasePolicy

class Policy(BasePolicy):
    # The overall policy class
    def __init__(self, opts):
        super(Policy, self).__init__(opts)
