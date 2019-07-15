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
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from data_loader import DataLoaderSimple as DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    # Agent options
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--memorize_views', type=str2bool, default=True)
    # Environment options
    parser.add_argument('--save_path', type=str, default='expert_trajectories.t7', help='Path to directory to save some sample results')

    opts = parser.parse_args()
    loaded_state = torch.load(opts.model_path)
    opts.init = 'xavier'
    opts.shuffle = False
    opts.mask_path = ''
    opts.reward_scale = 1e-2
    opts.start_views_json = ''
    opts.expert_rewards = False
    opts.supervised_scale = 1e-2
    opts.reward_scale_expert = 1e-4
    opts.expert_trajectories = False

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
    opts.actorType = loaded_state['opts'].actorType
    opts.baselineType = loaded_state['opts'].baselineType
    opts.act_full_obs = loaded_state['opts'].act_full_obs
    opts.mean_subtract = loaded_state['opts'].mean_subtract
    opts.wrap_azimuth = loaded_state['opts'].wrap_azimuth
    opts.wrap_elevation = loaded_state['opts'].wrap_elevation
    opts.critic_full_obs = loaded_state['opts'].critic_full_obs

    opts.A = opts.delta_M * opts.delta_N
    opts.P = opts.delta_M * opts.N

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)

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
    agent.policy.load_state_dict(loaded_state['state_dict'])
    train_trajectories = get_all_trajectories(loader, agent, 'train', opts)
    val_trajectories = get_all_trajectories(loader, agent, 'val', opts)
    test_trajectories = get_all_trajectories(loader, agent, 'test', opts)
    if opts.h5_path_unseen != '':
        test_unseen_trajectories = get_all_trajectories(loader, agent, 'test_unseen', opts)
        torch.save({'train': train_trajectories, 'val': val_trajectories, 'test': test_trajectories, 'test_unseen': test_unseen_trajectories}, open(opts.save_path, 'w')) 
    else:
        torch.save({'train': train_trajectories, 'val': val_trajectories, 'test': test_trajectories}, open(opts.save_path, 'w')) 
