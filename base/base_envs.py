from torch.autograd import Variable
from common import normalize_views

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

class BaseState(object):
    """
    Base class to simulates motion of an agent in 360 environments or 3D models.
    """

    def __init__(self, views, labs, views_rewards, start_idx, opts):
        """
            views: (B, N, M, C, H, W) numpy array
            labs: (B, *) numpy array
            views_rewards: (B, N, M) numpy array
            start_idx: Initial views for B panoramas [..., [e_idx, a_idx], ...]
        """
        # ---- Panorama navigation settings ----
        self.M = opts.M
        self.N = opts.N
        self.A = opts.A
        self.C = opts.num_channels
        self.start_idx = start_idx # Stored for the purpose of computing the loss
        self.idx = copy.deepcopy(start_idx) # Current view of the state
        # Whether elevation and azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Whether to wrap around elevation and azimuths
        self.wrap_elevation = opts.wrap_elevation
        self.wrap_azimuth = opts.wrap_azimuth

        # Decodes actions to the corresponding changes in elevation and azimuth
        self.act_to_delta = opts.act_to_delta
        self.delta_to_act = opts.delta_to_act

        # ---- Data settings ----
        self.batch_size = views.shape[0]
        self.delta = [[0, 0] for i in range(self.batch_size)] # Starts off with no change
        # Store mean and std to preprocess the views
        self.mean = opts.mean
        self.std = opts.std

        # ---- Save panorama data to the state ----
        self.views = np.copy(views)
        self.labs = labs
        if views_rewards is not None:
            self.views_rewards = np.copy(views_rewards)
            self.has_rewards = True
        else:
            self.views_rewards = np.zeros((views.shape[0], views.shape[1], views.shape[2]))
            self.has_rewards = False

        self.views_prepro = normalize_views(self.views, self.mean, self.std)

        # ---- Compute task-specific information ----
        self.precompute()

    def get_view(self, prepro=True):
        # Returns the current view and proprioception for each panorama
        # output view: BxCx32x32
        # output proprioception: list of [delta_elev, delta_azim, elev (optional), azim (optional)]

        pro_out = copy.deepcopy(self.delta)
        if self.knownElev or self.knownAzim:
            for i in range(len(pro_out)):
                if self.knownElev:
                    pro_out[i].append(self.idx[i][0])
                if self.knownAzim:
                    pro_out[i].append(self.idx[i][1])

        # Using python advanced indexing to get views for all panoramas simultaneously
        if prepro:
            views = self.views_prepro[range(len(self.idx)),
                                      [i[0] for i in self.idx],
                                      [i[1] for i in self.idx]]
        else:
            views = self.views[range(len(self.idx)),
                               [i[0] for i in self.idx],
                               [i[1] for i in self.idx]]

        return views, pro_out

    def rotate(self, act):
        """
        Rotates the state by delta corresponding to act. Returns the reward (intrinsic)
        corresponding to this transition.
        act: tensor of integers between 0 to opts.delta_M * opts.delta_N
        output reward: reward corresponding to visited view (optional)
        """

        N, M, bs, wpe, wpa = self.N, self.M, act.shape[0], self.wrap_elevation, self.wrap_azimuth
        d = [list(self.act_to_delta[act[i]]) for i in range(bs)]
        self.delta = d
        # Assume change in azimuth first, then elevation - to be consistent
        d_idx = [[d[i][0], d[i][1]] for i in range(bs)]

        # Elevation cannot be wrapped around
        if wpa:
            self.idx = [[np.clip(self.idx[i][0] + d_idx[i][0], 0, N-1),
                         (self.idx[i][1] + d_idx[i][1])%M] for i in range(bs)]
        else:
            self.idx = [[np.clip(self.idx[i][0] + d_idx[i][0], 0, N-1),
                         np.clip(self.idx[i][1] + d_idx[i][1], 0, M-1)] for i in range(bs)]

        # After reaching the next state, return the reward for this transition
        # Collect rewards and then zero them out.
        rewards_copy = np.copy(self.views_rewards[range(len(self.idx)), [i[0] for i in self.idx],
                                                                        [i[1] for i in self.idx]])
        if self.has_rewards: # To save some compute time
            for i in range(len(self.idx)):
                for j in range(self.idx[i][0]-1, self.idx[i][0]+2):
                    for k in range(self.idx[i][1]-1, self.idx[i][1]+2):
                        self.views_rewards[i, j%N, k%M] = 0
        return rewards_copy

    def precompute(self):
        """
        Perform task-specific pre-computation
        """
        pass

    def loss_fn(self, *args):
        """
        Compute task-specific loss
        """
        pass

    def reward_fn(self, *args):
        """
        Compute task-specific reward function
        """
        pass
