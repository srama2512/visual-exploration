"""
Script to evaluate pose-estimation policies
"""

import os
import sys
import pdb
import math
import json
import torch
import random
import argparse
import torchvision
import tensorboardX
import torch.optim as optim
import torchvision.utils as vutils

from envs import *
from utils import *
from agent import *
from base.common import *
from arguments import get_args
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    opts = get_args()

    opts.mask_path = ''
    opts.shuffle = False
    opts.init = 'xavier'
    opts.reward_scale = 1
    opts.start_views_json = ''
    opts.expert_rewards = False
    opts.reward_scale_expert = 1e-4

    loaded_state = torch.load(opts.model_path)

    opts.T = loaded_state['opts'].T
    opts.M = loaded_state['opts'].M
    opts.N = loaded_state['opts'].N
    opts.delta_M = loaded_state['opts'].delta_M
    opts.delta_N = loaded_state['opts'].delta_N
    opts.h5_path = loaded_state['opts'].h5_path
    opts.dataset = loaded_state['opts'].dataset
    opts.actOnElev = loaded_state['opts'].actOnElev
    opts.actOnAzim = loaded_state['opts'].actOnAzim
    opts.actOnTime = loaded_state['opts'].actOnTime
    opts.knownElev = loaded_state['opts'].knownElev
    opts.knownAzim = loaded_state['opts'].knownAzim
    opts.wrap_azimuth = loaded_state['opts'].wrap_azimuth
    opts.wrap_elevation = loaded_state['opts'].wrap_elevation
    opts.act_to_delta = loaded_state['opts'].act_to_delta
    opts.delta_to_act = loaded_state['opts'].delta_to_act
    opts.mean_subtract = loaded_state['opts'].mean_subtract
    if opts.actorType == 'unset':
        opts.actorType = loaded_state['opts'].actorType
    if opts.const_act == -1:
        if hasattr(loaded_state['opts'], 'const_act'):
            opts.const_act = loaded_state['opts'].const_act
    opts.baselineType = loaded_state['opts'].baselineType
    opts.act_full_obs = loaded_state['opts'].act_full_obs
    opts.critic_full_obs = loaded_state['opts'].critic_full_obs

    opts.A = opts.delta_M * opts.delta_N
    opts.P = opts.delta_M * opts.N

    # Defining the real-world angles corresponding the each elevation, azimuth index
    if opts.dataset == 0:
        # This is indexing on the reconstructed viewgrid which only
        # consists of the original viewgrid region
        opts.idx_to_angles = {}
        # Original viewgrid
        elev_idx_to_angle = [67.5, 22.5, -22.5, -67.5]
        azim_idx_to_angle = [-135, -90, -45, 0, 45, 90, 135,  180]
    elif opts.dataset == 1:
        opts.idx_to_angles = {}
        # Original viewgrid
        elev_idx_to_angle = [75, 45, 15, -15, -45, -75]
        azim_idx_to_angle = [20, 56, 92, 128, 164, 200, 236, 272, 308, 344]

    for e in range(opts.N):
        for a in range(opts.M):
            opts.idx_to_angles[(e, a)] = (math.radians(elev_idx_to_angle[e]),
                                          math.radians(azim_idx_to_angle[a]))

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    if opts.actorType == 'saved_trajectories' or opts.actorType == 'demo_sidekick' or opts.actorType == 'peek_saliency':
        from data_loader import DataLoaderExpertPolicy as DataLoader
    else:
        from data_loader import DataLoaderSimple as DataLoader

    if opts.dataset == 0:
        if opts.mean_subtract:
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [1, 1, 1]
        opts.num_channels = 3
    elif opts.dataset == 1:
        if opts.mean_subtract:
            opts.mean = [193.0162338615919]
            opts.std = [37.716024486312811]
        else:
            opts.mean = [0]
            opts.std = [0]
        opts.num_channels = 1
    else:
        raise ValueError('Dataset %d does not exist!'%(opts.dataset))

    loader = DataLoader(opts)
    agent = Agent(opts, mode='eval')
    agent.policy.load_state_dict(loaded_state['state_dict'], strict=False)

    if opts.eval_val:
        val_azim_err, val_elev_err = evaluate_pose_transfer(loader, agent, 'val', opts)
    else:
        test_azim_err, test_elev_err = evaluate_pose_transfer(loader, agent, 'test', opts)
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                test_unseen_azim_err, test_unseen_elev_err = evaluate_pose_transfer(loader, agent, 'test_unseen', opts)


    if opts.eval_val:
        print('========== Validation =========\nAzmiuth angular error || Elevation angular error\n{:15.5f},      {:15.5f}'.format(val_azim_err, val_elev_err))
    else:
        print('========== Test =========\nAzmiuth angular error || Elevation angular error\n{:15.5f}, {:15.5f}'.format(test_azim_err, test_elev_err))
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                print('========== Test (unseen) =========\nAzmiuth angular error || Elevation angular error\n{:15.5f}, {:15.5f}'.format(test_unseen_azim_err, test_unseen_elev_err))
