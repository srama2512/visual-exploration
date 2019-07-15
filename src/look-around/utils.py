import os
import cv2
import sys
import pdb
import math
import json
import torch
import random
import argparse
import numpy as np
import torchvision
import tensorboardX
import torch.optim as optim
import torchvision.utils as vutils

from envs import *
from PIL import Image
from base.common import *

def process_and_save(args):
    img = args[0]
    save_path = args[1]
    if img is not None:
        num_channels = img.shape[0]
        if num_channels == 1:
            img = Image.fromarray(img[0].astype(np.uint8)).convert('RGB')
        else:
            img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))
        img.save(save_path)

def load_module(agent, opts):
    """
    Given the agent, load a pre-trained model and other setup based on the
    training_setting
    """
    # ---- Load the pre-trained model ----
    load_state = torch.load(opts.load_model)
    # strict=False ensures that only the modules common to loaded_dict and agent.policy's state_dict are loaded.
    # Could potentially lead to errors being masked. Tread carefully!
    if opts.actorType == 'actor' and opts.act_full_obs:
        # Don't load the actor module, since the full obs actor architecture is different.
        partial_state_dict = {k: v for k, v in load_state['state_dict'].items() if 'act' not in k}
        agent.policy.load_state_dict(partial_state_dict, strict=False)
    else:
        agent.policy.load_state_dict(load_state['state_dict'], strict=False) 

    # ---- Other settings ----
    epoch_start = 0
    best_val_error = 100000
    train_history = []
    val_history = []

    if opts.training_setting == 1:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be 
        finetuned for multi-view reconstruction.
        """
        # (1) Must fix sense, fuse modules
        for parameter in agent.policy.sense_im.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.sense_pro.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.fuse.parameters():
            parameter.requires_grad = False
        # (2) Fix decode module if requested
        if opts.fix_decode:
            for parameter in agent.policy.decode.parameters():
                parameter.requires_grad = False
        # (3) Re-create the optimizer with the above settings
        agent.create_optimizer(opts.lr, opts.weight_decay, opts.training_setting, opts.fix_decode)

    elif opts.training_setting == 2:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be
        further trained on the same setting.
        """
        # (1) Keep a copy of the new number of epochs to run for
        epoch_total = opts.epochs
        # (2) Load the rest of the opts from saved model
        opts = load_state['opts']
        opts.epochs = epoch_total
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        # (3) Create optimizer based on the new parameter settings 
        agent.create_optimizer(opts.lr, opts.weight_decay, 2, opts.fix_decode)
        # (4) Load the optimizer state dict
        agent.optimizer.load_state_dict(load_state['optimizer'])

    elif opts.training_setting == 3:
        """
        Scenario: Model training on multi-view reconstruction. Needs to be 
        further trained on the same setting.
        """
        # (1) Load opts from saved model and replace LR
        opts_copy = load_state['opts']
        opts_copy.lr = opts.lr
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        opts_copy.training_setting = opts.training_setting
        opts = opts_copy
        # (2) Fix sense, fuse and decode (optionally) modules
        for parameter in agent.policy.sense_im.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.sense_pro.parameters():
            parameter.requires_grad = False
        for parameter in agent.policy.fuse.parameters():
            parameter.requires_grad = False
        if opts.fix_decode:
            for parameter in agent.policy.decode.parameters():
                parameter.requires_grad = False
        # (3) Re-create the optimizer with the above settings
        agent.create_optimizer(opts.lr, opts.weight_decay, 3, opts.fix_decode)
        # (4) Load the optimizer state dict
        agent.optimizer.load_state_dict(load_state['optimizer'])

    elif opts.training_setting == 4:
        """
        Scenario: Model trained on one-view reconstruction. Needs to be
        further trained on some other setting.
        """
        # (1) Load the train history, val history and best validation errors from the saved model.
        train_history = load_state['train_history']
        val_history = load_state['val_history']
        best_val_error = load_state['best_val_error']
        epoch_start = load_state['epoch']+1
        # (2) Create the optimizer according to the new settings
        agent.create_optimizer(opts.lr, opts.weight_decay, opts.training_setting, False)

    return best_val_error, train_history, val_history, epoch_start

def evaluate(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    overall_err = 0
    overall_count = 0
    err_values = []
    decoded_images = []
    while not depleted:
        # ---- Sample batch of data ----
        if opts.expert_trajectories or opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
            pano, pano_maps, depleted = loader.next_batch(split)
            pano_rewards = None
        elif opts.expert_rewards:
            pano, pano_rewards, depleted = loader.next_batch(split)
            pano_maps = None
        else:
            pano, depleted = loader.next_batch(split)
            pano_rewards, pano_maps = None, None

        # Initial setup for evaluating over a grid of views
        curr_err = 0
        curr_count = 0
        curr_err_batch = 0
        batch_size = pano.shape[0]
        # Compute the performance with the initial state
        # starting at fixed grid locations
        elevations = range(0, opts.N, 2)
        azimuths = range(0, opts.M, 2)

        elev_to_vis = elevations[random.randint(0, len(elevations)-1)]
        azim_to_vis = azimuths[random.randint(0, len(azimuths)-1)]

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})

                # For some random initial state, print the decoded images at all time steps
                if i == elev_to_vis and j == azim_to_vis:
                    curr_decoded_plus_true = None
                    for dec_idx in range(len(decoded_all)):
                        decoded = decoded_all[dec_idx].data.cpu()
                        curr_decoded = decoded.numpy()
                        # Rotate it forward by the start index
                        # Shifting all the images by equal amount since the start idx is same for all
                        elev_start = start_idx[0][0]
                        azim_start = start_idx[0][1]
                        if not opts.knownAzim:
                            curr_decoded = np.roll(curr_decoded, azim_start, axis=2)
                        if not opts.knownElev:
                            curr_decoded = np.roll(curr_decoded, elev_start, axis=1)

                        # Fill in the true views here
                        for jdx, jdx_v in enumerate(visited_idxes):
                            if jdx > dec_idx:
                                break
                            for idx in range(batch_size):
                                elev_curr = jdx_v[idx][0]
                                azim_curr = jdx_v[idx][1]
                                curr_decoded[idx, elev_curr, azim_curr, :, :, :] = state.views_prepro[idx, elev_curr, azim_curr, :, :, :]
                        curr_decoded = curr_decoded * 255
                        for c in range(opts.num_channels):
                            curr_decoded[:, :, :, c, :, :] += opts.mean[c]

                        if opts.num_channels == 1:
                            # convert to 3 channeled image by repeating grayscale
                            curr_decoded = np.repeat(curr_decoded, 3, axis=3)

                        jdx_v = visited_idxes[dec_idx]
                        for idx in range(batch_size):
                            # fill in some red margin
                            elev_curr = jdx_v[idx][0]
                            azim_curr = jdx_v[idx][1]
                            curr_decoded[idx, elev_curr, azim_curr, :, 0:3, :] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, -3:, :] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, :, 0:3] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, :, -3:] = 0
                            curr_decoded[idx, elev_curr, azim_curr, 0, 0:3, :] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, -3:, :] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, :, 0:3] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, :, -3:] = 255

                        # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
                        # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
                        curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)
                        true_state = np.array(state.views)
                        start_idx = state.start_idx
                        if opts.num_channels == 1:
                            # convert to 3 channeled image by repeating grayscale
                            true_state = np.repeat(true_state, 3, axis=3)

                        # Fill in red margin for starting states of each true panorama
                        for idx in range(batch_size):
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, 0:3, :] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, -3:, :] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, 0:3] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, -3:] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, 0:3, :] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, -3:, :] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, 0:3] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, -3:] = 255

                        true_state = true_state.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)

                        # Draw arrows representing the actions on the true_state
                        for idx in range(batch_size):
                            true_state_curr = true_state[idx, 0].transpose(1, 2, 0).copy()
                            for jdx in range(1, len(visited_idxes)):
                                elev_curr = visited_idxes[jdx][idx][0]
                                azim_curr = visited_idxes[jdx][idx][1]
                                elev_prev = visited_idxes[jdx-1][idx][0]
                                azim_prev = visited_idxes[jdx-1][idx][1]
                                arrow_start = (azim_prev * 32 + 16, elev_prev * 32 + 16)
                                arrow_end   = (azim_curr * 32 + 16, elev_curr * 32 + 16)
                                draw_arrow(true_state_curr, arrow_start, arrow_end, (255, 0, 0))
                            true_state[idx, 0] = true_state_curr.transpose(2, 0, 1)

                        if curr_decoded_plus_true is None:
                            curr_decoded_plus_true = curr_decoded
                        else:
                            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

                    curr_decoded_plus_true = np.concatenate([true_state, curr_decoded_plus_true], axis=1)
                    if opts.expert_rewards:
                        reward_image = np.zeros_like(true_state)
                        for iter_N in range(opts.N):
                            for iter_M in range(opts.M):
                                for bn in range(batch_size):
                                    reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
                        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

                    decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))

                # Add error from the last step
                curr_err += rec_errs[-1].data.sum()
                curr_count += 1 # Count for the views
                curr_err_batch += rec_errs[-1].data.cpu().numpy()

        curr_err /= curr_count
        curr_err_batch /= curr_count
        for i in range(curr_err_batch.shape[0]):
            err_values.append(float(curr_err_batch[i]))
        overall_err += curr_err
        overall_count += batch_size

    err_values = np.array(err_values)
    overall_mean = float(np.mean(err_values))
    overall_std = float(np.std(err_values, ddof=1))
    overall_std_err = float(overall_std/math.sqrt(err_values.shape[0]))

    agent.policy.train()

    return overall_mean, overall_std, overall_std_err, decoded_images

def evaluate_adversarial(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over all grid locations as
    starting points and returns the average of worst reconstruction error over different
    locations for the panoramas (average(max error over locations)).
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    overall_err = 0
    overall_count = 0
    decoded_images = []
    err_values = []
    while not depleted:
        # ---- Sample batch of data ----
        if opts.expert_trajectories or opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
            pano, pano_maps, depleted = loader.next_batch(split)
            pano_rewards = None
        elif opts.expert_rewards:
            pano, pano_rewards, depleted = loader.next_batch(split)
            pano_maps = None
        else:
            pano, depleted = loader.next_batch(split)
            pano_rewards = None
            pano_maps = None

        # Initial setup for evaluating over a grid of views
        batch_size = pano.shape[0]
        # Compute the performance with the initial state
        # starting at fixed grid locations

        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)

        errs_across_grid = np.zeros((batch_size, opts.N, opts.M))

        elev_to_vis = elevations[random.randint(0, len(elevations)-1)]
        azim_to_vis = azimuths[random.randint(0, len(azimuths)-1)]

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency': # Not enabling demo_sidekick training for AgentSupervised (that's not needed, doesn't make sense)
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, rec_errs, _, _,  _, _, visited_idxes, decoded_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # For some random initial state, print the decoded images at all time steps
                if i == elev_to_vis and j == azim_to_vis:
                    curr_decoded_plus_true = None
                    for dec_idx in range(len(decoded_all)):
                        decoded = decoded_all[dec_idx].data.cpu()
                        curr_decoded = decoded.numpy()
                        # Rotate it forward by the start index
                        # Shifting all the images by equal amount since the start idx is same for all
                        elev_start = start_idx[0][0]
                        azim_start = start_idx[0][1]
                        if not opts.knownAzim:
                            curr_decoded = np.roll(curr_decoded, azim_start, axis=2)
                        if not opts.knownElev:
                            curr_decoded = np.roll(curr_decoded, elev_start, axis=1)

                        # Fill in the true views here
                        for jdx, jdx_v in enumerate(visited_idxes):
                            if jdx > dec_idx:
                                break
                            for idx in range(batch_size):
                                elev_curr = jdx_v[idx][0]
                                azim_curr = jdx_v[idx][1]
                                curr_decoded[idx, elev_curr, azim_curr, :, :, :] = state.views_prepro[idx, elev_curr, azim_curr, :, :, :]
                        curr_decoded = curr_decoded * 255
                        for c in range(opts.num_channels):
                            curr_decoded[:, :, :, c, :, :] += opts.mean[c]

                        if opts.num_channels == 1:
                            # convert to 3 channeled image by repeating grayscale
                            curr_decoded = np.repeat(curr_decoded, 3, axis=3)

                        jdx_v = visited_idxes[dec_idx]
                        for idx in range(batch_size):
                            # Fill in some red margin
                            elev_curr = jdx_v[idx][0]
                            azim_curr = jdx_v[idx][1]
                            curr_decoded[idx, elev_curr, azim_curr, :, 0:3, :] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, -3:, :] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, :, 0:3] = 0
                            curr_decoded[idx, elev_curr, azim_curr, :, :, -3:] = 0
                            curr_decoded[idx, elev_curr, azim_curr, 0, 0:3, :] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, -3:, :] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, :, 0:3] = 255
                            curr_decoded[idx, elev_curr, azim_curr, 0, :, -3:] = 255

                        # Need to convert from B x N x M x C x 32 x 32 to B x 1 x C x N*32 x M*32
                        # Convert from B x N x M x C x 32 x 32 to B x C x N x 32 x M x 32 and then reshape
                        curr_decoded = curr_decoded.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)
                        true_state = np.array(state.views)
                        start_idx = state.start_idx
                        if opts.num_channels == 1:
                            # convert to 3 channeled image by repeating grayscale
                            true_state = np.repeat(true_state, 3, axis=3)

                        # Fill in red margin for starting states of each true panorama
                        for idx in range(batch_size):
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, 0:3, :] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, -3:, :] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, 0:3] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], :, :, -3:] = 0
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, 0:3, :] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, -3:, :] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, 0:3] = 255
                            true_state[idx, start_idx[idx][0], start_idx[idx][1], 0, :, -3:] = 255

                        true_state = true_state.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 1, 3, opts.N*32, opts.M*32)

                        if curr_decoded_plus_true is None:
                            curr_decoded_plus_true = curr_decoded
                        else:
                            curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, curr_decoded], axis=1)

                    curr_decoded_plus_true = np.concatenate([true_state, curr_decoded_plus_true], axis=1)
                    if opts.expert_rewards:
                        reward_image = np.zeros_like(curr_decoded)
                        for iter_N in range(opts.N):
                            for iter_M in range(opts.M):
                                for bn in range(batch_size):
                                    reward_image[bn, :, :, (iter_N*32):((iter_N+1)*32), (iter_M*32):((iter_M+1)*32)] = pano_rewards[bn, iter_N, iter_M]/255.0
                        curr_decoded_plus_true = np.concatenate([curr_decoded_plus_true, reward_image], axis=1)

                    decoded_images.append(torch.Tensor(curr_decoded_plus_true/255.0))
                # endif
                errs_across_grid[:, i, j] = rec_errs[-1].data.cpu().numpy()

        errs_across_grid = errs_across_grid.reshape(batch_size, -1)
        overall_err += np.sum(np.max(errs_across_grid, axis=1))
        overall_count += batch_size
        err_values.append(np.max(errs_across_grid, axis=1))

    err_values = np.concatenate(err_values, axis=0)
    overall_mean = np.mean(err_values)
    overall_std = np.std(err_values, ddof=1)
    overall_std_err = overall_std/math.sqrt(err_values.shape[0])

    agent.policy.train()

    return overall_mean, overall_std, overall_std_err, decoded_images 

def get_all_trajectories(loader, agent, split, opts):
    """
    Gathers trajectories from all starting positions and returns them.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    trajectories = {}

    # Sample all the locations from grid
    elevations = range(0, opts.N)
    azimuths = range(0, opts.M)

    for i in elevations:
        for j in azimuths:
            trajectories[(i, j)] = []

    while not depleted:
        # ---- Sample batch of data ----
        pano, depleted = loader.next_batch(split)
        pano_rewards, pano_maps = None, None

        batch_size = pano.shape[0]
        # Gather agent trajectories from each starting location
        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                _, _, _, _, _, _, _, _, actions_taken = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
                # actions_taken: B x T torch Tensor
                trajectories[(i, j)].append(actions_taken)

    for i in elevations:
        for j in azimuths:
            trajectories[(i, j)] = torch.cat(trajectories[(i, j)], dim=0)

    agent.policy.train()

    return trajectories

def get_all_decodings(loader, agent, split, opts, pool):
    """
    Gathers decoder outputs from all starting positions and returns them.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    decoded_panoramas = {}
    true_panoramas = {}

    cvt_viewgrid = lambda x: x.transpose((0, 3, 1, 4, 2, 5)).reshape(-1, opts.num_channels, opts.N*32, opts.M*32)
    while not depleted:
        # ---- Sample batch of data ----
        pano, depleted = loader.next_batch(split)
        pano_rewards, pano_maps = None, None

        batch_size = pano.shape[0]
        # Gather agent decoded_panoramas from each starting location
        # Sample some random location from grid for each element in batch
        all_elevations = range(0, opts.N)
        all_azimuths = range(0, opts.M)

        for sidx in range(opts.num_starts):
            elevations = [random.choice(all_elevations) for _ in range(batch_size)]
            azimuths   = [random.choice(all_azimuths) for _ in range(batch_size)]

            start_idx = [[i, j] for i, j in zip(elevations, azimuths)]
            state = State(pano, pano_rewards, start_idx, opts)
            # Enable view memorization for testing by default
            _, _, _, _, _, _, _, decoded_images, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
            decoded_images = decoded_images[-1].cpu().data.numpy()*255.0
            true_images = state.views_prepro_shifted.copy()*255.0

            # Undo mean subtraction
            for c in range(opts.num_channels):
                decoded_images[:, :, :, c, :, :] += opts.mean[c]
                true_images[:, :, :, c, :, :] += opts.mean[c]

            decoded_images = np.clip(decoded_images, 0.0, 255.0)
            true_images = np.clip(true_images, 0.0, 255.0)

            decoded_images = cvt_viewgrid(decoded_images.astype(np.uint8))
            true_images    = cvt_viewgrid(true_images.astype(np.uint8))

            src_imgs, tgt_imgs = zip(*[(decoded_images[b], true_images[b]) for b in range(batch_size)])
            current_count = opts.count
            src_paths, tgt_paths = zip(*[(os.path.join(opts.save_dir, 'A', split, '{:06d}.png'.format(cnt)), \
                                          os.path.join(opts.save_dir, 'B', split, '{:06d}.png'.format(cnt))) \
                                         for cnt in range(opts.count, opts.count+batch_size)])

            src_imgs  = list(src_imgs ); tgt_imgs  = list(tgt_imgs )
            src_paths = list(src_paths); tgt_paths = list(tgt_paths)

            if batch_size < opts.batch_size:
                src_imgs  += [None for _ in range(opts.batch_size - batch_size)]
                tgt_imgs  += [None for _ in range(opts.batch_size - batch_size)]
                src_paths += [None for _ in range(opts.batch_size - batch_size)]
                tgt_paths += [None for _ in range(opts.batch_size - batch_size)]

            _ = pool.map(process_and_save, zip(src_imgs, src_paths))
            _ = pool.map(process_and_save, zip(tgt_imgs, tgt_paths))

            opts.count += batch_size

    agent.policy.train()

def get_visualization_data(loader, agent, split, opts):
    """
    Gathers GT viewgrid, reconstructed viewgrid at each time step, agent's positions 
    on the viewgrid, reconstruction errors at each time step, M and N.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    gt_viewgrids  = []
    rec_viewgrids = []
    agent_positions = []
    rec_errors = []
    C = opts.num_channels
    M = opts.M
    N = opts.N
    T = opts.T

    cvt_viewgrid = lambda x: x.transpose((0, 3, 1, 4, 2, 5)).reshape(-1, C, opts.N*32, opts.M*32)
    while not depleted:
        # ---- Sample batch of data ----
        pano, depleted = loader.next_batch(split)
        pano_rewards, pano_maps = None, None

        batch_size = pano.shape[0]

        # Sample some random location from grid for each element in batch
        all_elevations = range(0, opts.N)
        all_azimuths = range(0, opts.M)

        elevations = [random.choice(all_elevations) for _ in range(batch_size)]
        azimuths   = [random.choice(all_azimuths) for _ in range(batch_size)]

        start_idx = [[i, j] for i, j in zip(elevations, azimuths)]
        state = State(pano, pano_rewards, start_idx, opts)
        # Enable view memorization for testing by default
        _, rec_errs, _, _, _, _, visited_idxes, decoded_images, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})
        # decoded_images[*].shape = (batch_size, N, M, C, 32, 32)
        decoded_images = np.concatenate([img.cpu().data.unsqueeze(1).numpy()*255.0 for img in decoded_images], axis=1)
        # decoded_images.shape = (batch_size, T, N, M, C, 32, 32)
        # true_images.shape = (batch_size, N, M, C, 32, 32)
        true_images = state.views_prepro_shifted.copy()*255.0

        # Shift the images based on the agent's starting position
        for b in range(batch_size):
            azim_start = visited_idxes[0][b][1]
            elev_start = visited_idxes[0][b][0]
            if not opts.knownAzim:
                decoded_images[b] = np.roll(decoded_images[b], azim_start, axis=2)
                true_images[b] = np.roll(true_images[b], azim_start, axis=1)
            if not opts.knownElev:
                decoded_images[b] = np.roll(decoded_images[b], elev_start, axis=1)
                true_images[b] = np.roll(true_images[b], elev_start, axis=0)

        # Undo mean subtraction
        for c in range(opts.num_channels):
            decoded_images[:, :, :, :, c, :, :] += opts.mean[c]
            true_images[:, :, :, c, :, :] += opts.mean[c]

        decoded_images = np.clip(decoded_images, 0.0, 255.0)
        true_images = np.clip(true_images, 0.0, 255.0)

        # Rotate the image by the starting azimuth

        shp = decoded_images.shape
        # decoded_images.shape = (batch_size, T, C, N*32, M*32)
        # true_images.shape = (batch_size, C, N*32, M*32)
        decoded_images = cvt_viewgrid(decoded_images.astype(np.uint8).reshape(-1, *shp[2:])).reshape(batch_size, T, C, 32*N, 32*M)
        true_images    = cvt_viewgrid(true_images.astype(np.uint8))

        # reconstruction errors across time
        # rec_errs.shape = (batch_size, T)
        rec_errs = np.concatenate([rec_err.cpu().data.unsqueeze(1).numpy() for rec_err in rec_errs], axis=1)

        # agent positions across time
        # visited_idxes - T-list of batch_size-list of 2-list
        visited_idxes = np.array(visited_idxes).transpose(1, 0, 2) # visited_idxes.shape = (T, batch_size, 2)

        rec_errors.append(rec_errs)
        gt_viewgrids.append(true_images)
        rec_viewgrids.append(decoded_images)
        agent_positions.append(visited_idxes)

        break

    rec_errors = np.concatenate(rec_errors, axis=0)
    gt_viewgrids = np.concatenate(gt_viewgrids, axis=0)
    rec_viewgrids = np.concatenate(rec_viewgrids, axis=0)
    agent_positions = np.concatenate(agent_positions, axis=0)

    agent.policy.train()

    return gt_viewgrids, rec_viewgrids, rec_errors, agent_positions
