"""
Script to train pose-estimation policies
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

    from data_loader import DataLoaderSimple as DataLoader

    if opts.dataset == 0:
        opts.num_channels = 3
        if opts.mean_subtract:
            # R, G, B means and stds
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [1, 1, 1]
    elif opts.dataset == 1:
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

    loader = DataLoader(opts)
    agent = Agent(opts)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path)
    # Set networks to train
    agent.policy.train()
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

        agent.policy.load_state_dict(loaded_model['state_dict'])
        train_history = loaded_model['train_history']
        val_history = loaded_model['val_history']
        #agent.optimizer.load_state_dict(loaded_model['optimizer'])
        best_val_error = loaded_model['best_val_error']

    # Some random selection of images to display
    rng_choices = random.sample(range(300//opts.batch_size), 2) 
    # Start training
    for epoch in range(epoch_start, opts.epochs):
        # Initialize epoch specific variables
        depleted = False
        train_azim_err = 0
        train_elev_err = 0
        train_count = 0
        iter_count = 0

        while not depleted:
            # pano - BxNxMxCx32x32
            pano, depleted = loader.next_batch('train')
            pano_rewards, pano_maps = None, None
            # Note: This batch size is the current batch size, not the global batch size. This varies

            # when you reach the boundary of the dataset.
            batch_size = pano.shape[0]
            start_idx = get_starts(opts.N, opts.M, batch_size, opts.start_view)
            state = State(pano, pano_rewards, start_idx, opts)
            # Forward pass
            log_probs, rewards, entropies, decoded, values, visited_idxes, decoded_all, _, mean_elev_error, mean_azim_error = agent.gather_trajectory(state, eval_opts=None, pano_maps=pano_maps, opts=opts)
            # Backward pass
            # Dummy errors since there are no task-prediction errors here
            task_errs = [Variable(torch.zeros(batch_size, )) for _ in range(opts.T)]
            if opts.iscuda:
                task_errs = [err.cuda() for err in task_errs]

            agent.update_policy(rewards, log_probs, task_errs, entropies, values) 

            # Accumulate statistics
            train_azim_err += mean_azim_error * batch_size
            train_elev_err += mean_elev_error * batch_size
            train_count += batch_size
            iter_count += 1

        train_azim_err /= train_count
        train_elev_err /= train_count

        # Evaluate the agent after every epoch
        val_azim_err, val_elev_err, decoded_images = evaluate_pose(loader, agent, 'val', opts)

        # Write out statistics to tensorboard
        writer.add_scalar('data/train_azim_error', train_azim_err, epoch+1)
        writer.add_scalar('data/train_elev_error', train_elev_err, epoch+1)
        writer.add_scalar('data/val_azim_error', val_azim_err, epoch+1)
        writer.add_scalar('data/val_elev_error', val_elev_err, epoch+1)

        # Write out models and other statistics to torch format file
        train_err = train_azim_err
        val_err = val_azim_err
        train_history.append([epoch, train_err])
        val_history.append([epoch, val_err])
        if best_val_error > val_err:
            best_val_error = val_err
            save_state = {
                            'epoch': epoch,
                            'state_dict': agent.policy.state_dict(),
                            'optimizer': agent.optimizer.state_dict(),
                            'opts': opts, 
                            'best_val_error': best_val_error,
                            'train_history': train_history,
                            'val_history': val_history
                         }
            torch.save(save_state, os.path.join(opts.save_path, 'model_best.net'))

        save_state = {
                        'epoch': epoch,
                        'state_dict': agent.policy.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'opts': opts,
                        'best_val_error': best_val_error,
                        'train_history': train_history,
                        'val_history': val_history
                     }
        torch.save(save_state, os.path.join(opts.save_path, 'model_latest.net'))

        print('Epoch %d : Train loss: %9.6f    Val loss: %9.6f'%(epoch+1, train_err, val_err))

        # Display three randomly selected batches of panoramas every 10 epochs
        if (epoch+1) % 10 == 0 or epoch == 0:
            for choice in rng_choices:
                for pano_count in range(decoded_images[choice].size(0)):
                    x = vutils.make_grid(decoded_images[choice][pano_count], padding=5, normalize=True, scale_each=True, nrow=opts.T//2+1)
                    writer.add_image('Validation batch # : %d  image # : %d'%(choice, pano_count), x, 0) # Converting this to 0 to save disk space, should be epoch ideally

if __name__ == '__main__':

    opts = get_args()

    train(opts)
