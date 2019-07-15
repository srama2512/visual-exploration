import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.autograd import Variable
from base.base_policy import BasePolicy
from base.common import *

class Policy(BasePolicy):
    def __init__(self, opts):
        super(Policy, self).__init__(opts)

        # (6) Decode: Decodes aggregated views to panorama
        decode_layers = [
                            nn.BatchNorm1d(self.rnn_hidden_size),
                            nn.Linear(self.rnn_hidden_size, 1024),
                            nn.LeakyReLU(0.2, inplace=True),
                            View(-1, 64, 4, 4), # Bx64x4x4
                            nn.ConvTranspose2d(64, 256, kernel_size=5, stride=2,
                                               padding=2, output_padding=1), # Bx256x8x8
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2,
                                               padding=2, output_padding=1), # Bx128x16x16
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.ConvTranspose2d(128, self.M*self.N*self.C,
                                               kernel_size=5, stride=2, padding=2, output_padding=1) # B x M*(N/2)*C x 32 x 32
                         ]

        # If mean subtraction is performed, output can be negative
        if opts.mean_subtract:
            decode_layers.append(nn.LeakyReLU(0.8, inplace=True))
        else:
            decode_layers.append(nn.ReLU(inplace=True))
        self.decode = nn.Sequential(*decode_layers)

        # ---- Initialize parameters according to specified strategy ----
        if opts.init == 'xavier':
            init_strategy = ixvr
        elif opts.init == 'normal':
            init_strategy = inrml
        else:
            init_strategy = iunf

        self.decode = initialize_sequential(self.decode, init_strategy)

    def forward(self, x, hidden=None):
        probs, hidden, values = super(Policy, self).forward(x, hidden)

        # ---- Deocode the aggregated state ----
        # hidden[0] gives num_layer x batch_size x hidden_size , hence hidden[0][0] since
        # only one hidden layer is used
        batch_size = hidden[0][0].shape[0]
        x = self.decode(F.normalize(hidden[0][0], p=1, dim=1))
        decoded = x.view(batch_size, self.N, self.M, self.C, 32, 32)

        return probs, hidden, decoded, values
