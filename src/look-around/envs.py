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
    def __init__(self, views, views_rewards, start_idx, opts):
        labs = None # No labels for this task
        super(State, self).__init__(views, labs, views_rewards, start_idx, opts)

    def precompute(self):
        """
        Compute total_pixels and shifted viewgrids to enable fast reconstruction
        loss computation.
        """
        vshape = self.views.shape
        self.total_pixels = self.M * self.N * vshape[3] * vshape[4] * vshape[5]
        # Shift each panorama in views_prepro according to corresponding start_idxes
        B, N, M, C, H, W = self.views_prepro.shape
        start_idx = self.start_idx
        self.views_prepro_shifted = np.zeros((B, N, M, C, H, W))
        views_subset = np.copy(self.views_prepro)

        for i in range(self.batch_size):
            elev_start = start_idx[i][0]
            azim_start = start_idx[i][1]

            if not (self.knownElev or self.knownAzim):
                # Shift the azimuth and elevation
                self.views_prepro_shifted[i] = np.roll(np.roll(views_subset[i], -azim_start, axis=1), -elev_start, axis=0)
            elif not self.knownAzim:
                # Shift the azimuths
                self.views_prepro_shifted[i] = np.roll(views_subset[i], -azim_start, axis=1)
            elif not self.knownElev:
                # Shift the elevations
                self.views_prepro_shifted[i] = np.roll(views_subset[i], -elev_start, axis=0)

    def loss_fn(self, rec_views, iscuda):
        """
        Computes reconstruction loss between self.views_prepro_shifted and rec_views.
        rec_views: (B, N, M, C, H, W) torch Variable with preprocessed values
        masks: (B, N, M, C, H, W) torch Variable
        """
        B = self.batch_size
        true_views = Variable(torch.Tensor(self.views_prepro_shifted))
        if iscuda:
            true_views = true_views.cuda()

        loss = ((true_views - rec_views)**2).view(B, -1).sum(dim=1)/(self.total_pixels)

        return loss
