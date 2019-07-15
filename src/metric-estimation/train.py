"""
Script to train metric-estimation policy
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

from utils import *
from envs import *
from agent import *
from base.common import *
from arguments import get_args
from tensorboardX import SummaryWriter
from data_loader import DataLoaderSimple as DataLoader

def train(opts):
    """
    Training function - trains an agent for a fixed number of epochs
    """
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N
    # Set random seeds
    set_random_seeds(opts.seed)  
    # Create actions mapping
    count_act = 0
    opts.act_to_delta = {}
    opts.delta_to_act = {}
    for i in range(-(opts.delta_N//2), opts.delta_N//2+1):
        for j in range(-(opts.delta_M//2), opts.delta_M//2+1):
            opts.act_to_delta[count_act] = (i, j)
            opts.delta_to_act[(i, j)] = count_act
            count_act += 1

    if opts.dataset == 1:
        opts.num_channels = 1
        if opts.mean_subtract:
            # R, G, B means and stds
            opts.mean = [193.0162338615919]
            opts.std = [37.716024486312811]
        else:
            opts.mean = [0]
            opts.std = [1]
    else:
        raise ValueError('Dataset %d does not exist!'%(opts.dataset))

    loader = DataLoader(opts)
    agent = Agent(opts)

    if opts.load_model != '':
        loaded_model = torch.load(opts.load_model)
        agent.load_state_dict(loaded_model['state_dict'], strict=False)
        print('Loading pre-trained model from {}'.format(opts.load_model))

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path)
    # Set networks to train
    agent.train()
    # Initiate statistics storage variables
    best_val_error = 100000
    train_history = []
    val_history = []
    epoch_start = 0

    # To handle job eviction and restarts
    if os.path.isfile(os.path.join(opts.save_path, 'model_latest.net')):
        print('====> Resuming training from previous checkpoint')
        # undo most of the loading done before
        loaded_model = torch.load(os.path.join(opts.save_path, 'model_latest.net'))
        opts = loaded_model['opts']
        epoch_start = loaded_model['epoch'] + 1

        loader = DataLoader(opts)
        agent = Agent(opts) 

        agent.load_state_dict(loaded_model['state_dict'])
        train_history = loaded_model['train_history']
        val_history = loaded_model['val_history']
        best_val_error = loaded_model['best_val_error']

    # Some random selection of images to display
    rng_choices = random.sample(range(loader.counts['val']//opts.batch_size), 2) 
    # Start training
    for epoch in range(epoch_start, opts.epochs):
        # Initialize epoch specific variables
        depleted = False
        train_err = 0
        train_count = 0
        iter_count = 0

        while not depleted:
            # pano - BxNxMxCx32x32
            pano, pano_labs, depleted = loader.next_batch('train')
            pano_rewards = None
            pano_maps = None
            pano_labs = Variable(torch.Tensor(pano_labs))
            if opts.iscuda:
                pano_labs = pano_labs.cuda()
            # Note: This batch size is the current batch size, not the global batch size. This varies

            # when you reach the boundary of the dataset.
            batch_size = pano.shape[0]
            start_idx = get_starts(opts.N, opts.M, batch_size, opts.start_view)
            state = State(pano, pano_labs, pano_rewards, start_idx, opts)
            # Forward pass
            log_probs, pred_errs, rewards, entropies, values, visited_idxes, predictions_all, _ = agent.gather_trajectory(state, eval_opts=None, pano_maps=pano_maps, opts=opts)
            # Backward pass
            agent.update_policy(rewards, log_probs, pred_errs, entropies, values)

            # Accumulate statistics
            train_err += pred_errs[-1].data.sum()
            train_count += batch_size
            iter_count += 1

        train_err /= train_count

        # Evaluate the agent after every epoch
        val_err, val_scores_across_time, vis_images = evaluate(loader, agent, 'val', opts)

        # Write out statistics to tensorboard
        writer.add_scalar('data/train_error', train_err, epoch+1)
        writer.add_scalar('data/val_error', val_err, epoch+1)
        writer.add_scalar('data/val_scores', val_scores_across_time[-1], epoch+1)

        # Write out models and other statistics to torch format file
        train_history.append([epoch, train_err])
        val_history.append([epoch, val_err])
        save_state = {
                        'epoch': epoch,
                        'state_dict': agent.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'opts': opts, 
                        'best_val_error': best_val_error,
                        'train_history': train_history,
                        'val_history': val_history
                     }
        if best_val_error > val_err:
            best_val_error = val_err
            save_state['best_val_err'] = best_val_error
            torch.save(save_state, os.path.join(opts.save_path, 'model_best.net'))

        torch.save(save_state, os.path.join(opts.save_path, 'model_latest.net'))

        print('Epoch %d : Train loss: %9.6f    Val loss: %9.6f    Val score: %9.6f'%(epoch+1, train_err, val_err, val_scores_across_time[-1]))

        # Display three randomly selected batches of panoramas every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            for choice in rng_choices:
                for pano_count in range(vis_images[choice].size(0)):
                    x = vutils.make_grid(vis_images[choice][pano_count], padding=5, normalize=False, scale_each=False, nrow=opts.T//2+1) 
                    writer.add_image('Validation batch # : %d  image # : %d'%(choice, pano_count), x, 0) # Converting this to 0 to save disk space, should be epoch ideally

if __name__ == '__main__':

    opts = get_args()

    opts.reward_scale_expert = 1.0

    train(opts)
