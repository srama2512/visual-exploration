"""
Script to evaluate look-around policies
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
from tensorboardX import SummaryWriter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    # Agent options
    parser.add_argument('--T', type=int, default=-1)
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--memorize_views', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='unset', help='[ actor | random | saved_trajectories | const_action | peek_saliency ]')
    parser.add_argument('--const_act', type=int, default=-1, help='constant action to execute under const_action')
    # Environment options
    parser.add_argument('--start_view', type=int, default=0, help='[0 - random, 1 - center, 2 - alternate positions, 3 - adversarial]')
    parser.add_argument('--save_path', type=str, default='', help='Path to directory to save some sample results')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')
    parser.add_argument('--trajectories_type', type=str, default='utility_maps', help='[utility_maps | expert_trajectories | saliency_scores]')
    parser.add_argument('--eval_val', type=str2bool, default=False, help='Evaluate on validation set?')

    opts = parser.parse_args()
    opts.mask_path = ''
    opts.shuffle = False
    opts.init = 'xavier'
    opts.reward_scale = 1
    opts.start_views_json = ''
    opts.expert_rewards = False
    opts.supervised_scale = 1e-2
    opts.reward_scale_expert = 1e-4
    opts.expert_trajectories = False

    loaded_state = torch.load(opts.model_path)

    if opts.T == -1:
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

    if opts.start_view != 3:
        if opts.eval_val:
            val_err, val_std, val_std_err, _ = evaluate(loader, agent, 'val', opts)
        else:
            test_err, test_std, test_std_err, decoded_images = evaluate(loader, agent, 'test', opts)
            if opts.dataset == 1:
                if opts.h5_path_unseen != '':
                    test_unseen_err, test_unseen_std, test_unseen_std_err, decoded_images_unseen = evaluate(loader, agent, 'test_unseen', opts)

    else:
        if opts.eval_val:
            val_err, val_std, val_std_err, _ = evaluate(loader, agent, 'val', opts)
        else:
            test_err, test_std, test_std_err, decoded_images = evaluate_adversarial(loader, agent, 'test', opts)
            if opts.dataset == 1 and opts.h5_path_unseen != '':
                test_unseen_err, test_unseen_std, test_unseen_std_err, decoded_images_unseen = evaluate_adversarial(loader, agent, 'test_unseen', opts)

    if not opts.eval_val:
        writer = SummaryWriter(log_dir=opts.save_path)
        count_choice = min(loader.counts['test'] // opts.batch_size, 10)
        rng_choices = random.sample(range(loader.counts['test']//opts.batch_size), count_choice)
        for choice in rng_choices:
            for pano_count in range(decoded_images[choice].size(0)):
                x = vutils.make_grid(decoded_images[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T+1, pad_value=1.0)
                writer.add_image('Test batch #%d, image #%d'%(choice, pano_count), x, 0)
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                count_choice = min(loader.counts['test_unseen'] // opts.batch_size, 10)
                rng_choices = random.sample(range(loader.counts['test_unseen']//opts.batch_size), count_choice)
                for choice in rng_choices:
                    for pano_count in range(decoded_images_unseen[choice].size(0)):
                        x = vutils.make_grid(decoded_images_unseen[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T+1, pad_value=1.0)
                        writer.add_image('Test unseen batch #%d, image #%d'%(choice, pano_count), x, 0)
    if opts.eval_val:
        print('Val mean(x1000): %6.3f | std(x1000): %6.3f | std err(x1000): %6.3f'%(val_err*1000, val_std*1000, val_std_err*1000))
    else:
        print('===== Test error =====')
        print('%6.3f'%(test_err * 1000))
        if opts.dataset == 1:
            if opts.h5_path_unseen != '':
                print('===== Test unseen error =====')
                print('%6.3f'%(test_unseen_err * 1000))

        writer.close()
