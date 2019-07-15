"""
Script to evaluate light-source-localization policies
"""

import os
import sys
import pdb
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

    opts.shuffle = False
    opts.init = 'xavier'
    opts.reward_scale = 1
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
    opts.labels_path = loaded_state['opts'].labels_path
    opts.wrap_azimuth = loaded_state['opts'].wrap_azimuth
    opts.wrap_elevation = loaded_state['opts'].wrap_elevation
    opts.act_to_delta = loaded_state['opts'].act_to_delta
    opts.delta_to_act = loaded_state['opts'].delta_to_act
    opts.mean_subtract = loaded_state['opts'].mean_subtract
    if opts.actorType == 'unset':
        opts.actorType = loaded_state['opts'].actorType
    opts.baselineType = loaded_state['opts'].baselineType
    opts.act_full_obs = loaded_state['opts'].act_full_obs
    opts.critic_full_obs = loaded_state['opts'].critic_full_obs

    opts.A = opts.delta_M * opts.delta_N
    opts.P = opts.delta_M * opts.N

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

    if opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
        from data_loader import DataLoaderExpertPolicy as DataLoader
    else:
        from data_loader import DataLoaderSimple as DataLoader

    if opts.dataset == 1:
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
    agent.load_state_dict(loaded_state['state_dict'], strict=False)

    if opts.eval_val:
        val_err, val_scores_across_time, _ = evaluate(loader, agent, 'val', opts)
    else:
        test_err, test_scores_across_time, vis_images = evaluate(loader, agent, 'test', opts)

    if not opts.eval_val:
        writer = SummaryWriter(log_dir=opts.save_path)
        count_choice = min(loader.counts['test'] // opts.batch_size, 10)
        rng_choices = random.sample(range(loader.counts['test']//opts.batch_size), count_choice)
        for choice in rng_choices:
            for pano_count in range(vis_images[choice].size(0)):
                x = vis_images[choice][pano_count]
                writer.add_image('Test batch #%d, image #%d'%(choice, pano_count), x, 0)
        print('Test scores (RMSE/Acc):')
        print(','.join(['{:.5f}'.format(score) for score in test_scores_across_time]))
        writer.close()
    else:
        print('Val scores (RMSE/Acc):')
        print(','.join(['{:.5f}'.format(score) for score in val_scores_across_time]))
