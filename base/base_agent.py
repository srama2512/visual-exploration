from torch.autograd import Variable
from utils import *
from envs import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

class BaseAgent(object):
    """
    This agent implements the policy from Policy class and uses REINFORCE / Actor-Critic
    for policy improvement
    """
    def __init__(self, opts, mode='train'):
        # ---- Create the policy network ----
        self._create_policy(opts)
        # ---- Create the task-head (if any) ----
        self._create_task_head(opts)
        # ---- Panorama operation settings ----
        self.C = opts.num_channels
        self.T = opts.T
        self.M = opts.M
        self.N = opts.N
        # Whether elevation, azimuth or time are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Action extents
        self.delta_M = opts.delta_M
        self.delta_N = opts.delta_N
        # Preprocessing
        self.mean = opts.mean
        self.std = opts.std

        # ---- Optimization settings ----
        self.mode = mode
        self.actorType = opts.actorType
        self.baselineType = opts.baselineType
        self.act_full_obs = opts.act_full_obs
        self.critic_full_obs = opts.critic_full_obs
        self.iscuda = opts.iscuda
        if opts.iscuda:
            self.cuda()
        # Average reward baselines
        if self.baselineType == 'average' and mode == 'train':
            # Set a baseline for REINFORCE
            self.R_avg = 0
            self.R_avg_expert = 0
            # Average counts maintained to update baselines
            self.avg_count = 0
            self.avg_count_expert = 0
        # Scaling factors
        if self.mode == 'train':
            self.critic_coeff = opts.critic_coeff
            self.lambda_entropy = opts.lambda_entropy # Entropy term coefficient
        self.reward_scale = opts.reward_scale
        self.reward_scale_expert = opts.reward_scale_expert

        # ---- Create the optimizer ----
        if self.mode == 'train':
            self.create_optimizer(opts.lr, opts.weight_decay)

    def gather_trajectory(self, *args, **kwargs):
        """
        Gather trajectories over provided environments.
        """
        pass

    def update_policy(self, rewards, log_probs, task_errs, entropies, values=None):
        """
        This function will take the rewards, log probabilities and task-spencific errors from
        the trajectory and perform the parameter updates for the policy using
        REINFORCE / Actor-Critic.
        INPUTS:
            rewards: list of T-1 Tensors containing reward for each batch at each time step
            log_probs: list of T-1 logprobs Variables of each transition of batch
            task_errs: list of T error Variables for each transition of batch
            entropies: list of T-1 entropy Variables for each transition of batch
            values: list of T-1 predicted values Variables for each transition of batch
        """
        # ---- Setup initial values ----
        batch_size = task_errs[0].size(0)
        R = torch.zeros(batch_size) # Reward accumulator
        B = 0 # Baseline accumulator - used primarily for the average baseline case
        loss = Variable(torch.Tensor([0]))
        if self.iscuda:
            loss = loss.cuda()
            R = R.cuda()

        # ---- Task-specific error computation
        for t in reversed(range(self.T)):
            loss = loss + task_errs[t].sum()/batch_size

        # --- REINFORCE / Actor-Critic loss based on T-1 transitions ----
        # Note: This will automatically be ignored when self.T = 1
        for t in reversed(range(self.T-1)):
            if self.policy.actorType == 'actor':
                R = R + rewards[t] # A one sample MC estimate of Q[t]
                # Compute the advantage
                if self.baselineType == 'critic':
                    adv = R - values[t].data
                else:
                    if t == self.T-2:
                        B += self.R_avg
                    B += self.R_avg_expert * self.reward_scale_expert
                    adv = R - B
                # PG loss
                loss_term_1 = - (log_probs[t]*self.reward_scale*Variable(adv, requires_grad=False)).sum()/batch_size 
                # Entropy loss, maximize entropy
                loss_term_2 = - self.lambda_entropy*entropies[t].sum()/batch_size
                # Critic prediction error
                if self.baselineType == 'critic':
                    loss_term_3 = self.critic_coeff*((Variable(R, requires_grad=False) - values[t])**2).sum()/batch_size
                else:
                    loss_term_3 = 0

                loss = loss + loss_term_1 + loss_term_2 + loss_term_3

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

    def create_optimizer(self, lr, weight_decay):
        """
        Create the optimizer as self.optimizer
        """
        self.optimizer = optim.Adam(self._get_parameters(), lr=lr, weight_decay=weight_decay)

    def cuda(self):
        """
        Convert all relevant attributes to cuda
        """
        pass

    def train(self):
        """
        Set all relevant attributes to train()
        """
        pass

    def eval(self):
        """
        Set all relevant attributes to eval()
        """
        pass

    def _create_policy(self, opts):
        """
        Create task-specific policy as self.policy
        """
        pass

    def _create_task_head(self, opts):
        """
        Create task-head as self.task_head
        """
        pass

    def _get_parameters(self):
        """
        Return model parameters relevant to optimizer
        """
        pass
