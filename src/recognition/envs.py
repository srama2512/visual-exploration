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
    """
    Simulates motion of an agent in 360 environments or 3D models. The inputs are
    image features instead of images.
    """

    def __init__(self, views, labs, views_rewards, start_idx, opts): 
        """
            views: (B, N, M, F)  array
            labs: (B, ) numpy array
            views_rewards: (B, N, M) numpy array
            start_idx: Initial views for B panoramas [..., [e_idx, a_idx], ...]
        """
	# ---- Panorama navigation settings ----
        self.M = opts.M
        self.N = opts.N
        self.A = opts.A
        self.start_idx = start_idx # Stored for the purpose of computing the loss
        self.idx = copy.deepcopy(start_idx) # Current view of the state
        # Whether elevation and azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        # Whether to wrap around elevation and azimuths
        self.wrap_elevation = opts.wrap_elevation
        self.wrap_azimuth = opts.wrap_azimuth
        # Decodes actions to the corresponding changes in elevation and azimuth
        self.act_to_delta = opts.act_to_delta
        self.delta_to_act = opts.delta_to_act

        # ---- Data settings ----
        self.batch_size = views.shape[0]
        self.delta = [[0, 0] for i in range(self.batch_size)] # Starts off with no change

        # ---- Save panorama data to the state ----
        self.views = np.copy(views)
        self.labs = labs
        if views_rewards is not None:
            self.views_rewards = np.copy(views_rewards)
            self.has_rewards = True
        else:
            self.views_rewards = np.zeros((views.shape[0], views.shape[1], views.shape[2]))
            self.has_rewards = False

    def get_view(self):
        # Returns the current view, delta and proprioception for each panorama
        # output view: BxF
        # output delta: list of [delta_elev, delta_azim]
        # output proprioception: list of [elev, azim]

        delta_out = copy.deepcopy(self.delta)
        pro_out  = copy.deepcopy(self.idx)
        # Using python advanced indexing to get views for all panoramas simultaneously
        views = self.views[range(len(self.idx)),
                           [i[0] for i in self.idx],
                           [i[1] for i in self.idx]]

        return views, delta_out, pro_out

    def reward_fn(self, prediction_probs):
        """
        Given the predicted probabilities, the reward is 1 if the correct
        class has the highest probability.
        predicted_probs: numpy array (batch_size, num_classes)
        """
        out_rewards = np.zeros((self.batch_size,))
        out_rewards[self.labs == np.argmax(prediction_probs, axis=1)] = 1
        return out_rewards
