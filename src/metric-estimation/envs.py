from torch.autograd import Variable

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

from base.base_envs import BaseState

class State(BaseState):

    def loss_fn(self, predictions):
        return ((predictions - self.labs)/(self.labs + 1e-8))**2
