"""
Script to evaluate recognition policies
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
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

def eval(opts):
    """
    Evaluating function - evaluates on validation and test splits
    """
    
    # Load the previous state
    loaded_state = torch.load(opts.model_path)

    opts.h5_path = loaded_state['opts'].h5_path
    opts.dataset = loaded_state['opts'].dataset
    opts.combineDropout = loaded_state['opts'].combineDropout
    opts.featDropout = loaded_state['opts'].featDropout
    opts.addExtraLinearFuse = loaded_state['opts'].addExtraLinearFuse
    opts.T = loaded_state['opts'].T
    opts.M = loaded_state['opts'].M
    opts.N = loaded_state['opts'].N
    opts.F = loaded_state['opts'].F
    opts.dataset = loaded_state['opts'].dataset
    if opts.delta_M == -1:
        if hasattr(loaded_state['opts'], 'delta_M'):
            opts.delta_M = loaded_state['opts'].delta_M
            opts.delta_N = loaded_state['opts'].delta_N
        else:
            opts.delta_M = 5
            opts.delta_N = 5
    opts.rnn_type = loaded_state['opts'].rnn_type
    opts.actOnElev = loaded_state['opts'].actOnElev
    opts.actOnAzim = loaded_state['opts'].actOnAzim
    opts.actOnTime = loaded_state['opts'].actOnTime
    if opts.actorType == 'unset':
        opts.actorType = loaded_state['opts'].actorType
    opts.num_classes = loaded_state['opts'].num_classes 
    opts.wrap_azimuth = loaded_state['opts'].wrap_azimuth
    opts.act_to_delta = loaded_state['opts'].act_to_delta
    opts.delta_to_act = loaded_state['opts'].delta_to_act
    opts.baselineType = loaded_state['opts'].baselineType
    opts.wrap_elevation = loaded_state['opts'].wrap_elevation
    opts.act_to_delta = loaded_state['opts'].act_to_delta
    opts.delta_to_act = loaded_state['opts'].delta_to_act
    opts.optimizer_type = loaded_state['opts'].optimizer_type

    if hasattr(loaded_state['opts'], 'normalize_hidden'):
        opts.normalize_hidden = loaded_state['opts'].normalize_hidden
    else:
        opts.normalize_hidden = True
    if hasattr(loaded_state['opts'], 'nonlinearity'): 
        opts.nonlinearity = loaded_state['opts'].nonlinearity
    else:
        opts.nonlinearity = 'tanh'

    opts.init = 'xavier'
    opts.shuffle = 'False'
    opts.seed = 123
    opts.reward_scale = 1.0
    opts.reward_scale_expert = 1e-2
    opts.rewards_greedy = False
    opts.A = opts.delta_M * opts.delta_N

    # Set random seeds
    set_random_seeds(opts.seed)  
    # Data loading
    if opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
        from data_loader import DataLoaderExpertPolicy as DataLoader
    else:
        from data_loader import DataLoaderSimple as DataLoader

    loader = DataLoader(opts)
    opts.frequent_class = loader.get_most_frequent_class()

    # Create the agent
    agent = Agent(opts, mode='eval') 
    agent.policy.load_state_dict(loaded_state['policy_state_dict'], strict=False)
    for t in range(opts.T):
        agent.classifiers[t].load_state_dict(loaded_state['classifier_state_dict'][t])

    # Set networks to evaluate
    agent.policy.eval()
    for t in range(opts.T):
        agent.classifiers[t].eval()

    if opts.eval_val:
        if opts.compute_all_times:
            val_accuracy, val_accuracy_all_times, _, _ = evaluate(loader, agent, 'val', opts)
            print('====> Validation accuracy (vs) time')
            print(','.join(['{:.3f}'.format(_x * 100) for _x in val_accuracy_all_times]))
        else:
            val_accuracy, _, _ = evaluate(loader, agent, 'val', opts)   
        print('Validation accuracy: %.3f'%(val_accuracy*100))
    else:
        if opts.compute_all_times:
            test_accuracy, test_accuracy_all_times, _, _ = evaluate(loader, agent, 'test', opts)
            print('====> Testing accuracy (vs) time')
            print(','.join(['{:.3f}'.format(_x * 100) for _x in test_accuracy_all_times]))
        else:
            test_accuracy, _, _ = evaluate(loader, agent, 'test', opts)

        print('Testing accuracy: %.3f'%(test_accuracy*100))

if __name__ == '__main__':

    opts = get_args()
    opts.expert_rewards = False

    eval(opts)
