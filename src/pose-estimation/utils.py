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

def norm_angles(ags):
    # ags - angles in radians (batch_size, ) array
    return np.arctan2(np.sin(ags), np.cos(ags))

def evaluate_pose_transfer(loader, agent, split, opts):
    """
    Evaluation of reconstruction agent on pose task
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    azim_errors_all = []
    elev_errors_all = []
    while not depleted:
        # ---- Sample batch of data ----
        if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
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
        if opts.start_view == 0:
            # Randomly sample one location from grid
            elevations = [random.randint(0, opts.N-1)]
            azimuths = [random.randint(0, opts.M-1)]
        elif opts.start_view == 1:
            # Sample only the center location from grid
            elevations = [opts.N//2]
            azimuths = [opts.M//2-1]
        elif opts.start_view == 2:
            elevations = range(0, opts.N, 2)
            azimuths = range(0, opts.M, 2)

        valid_out_elevations = np.array(range(0, opts.N))
        valid_out_azimuths = np.array(range(0, opts.M))

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
                    _, rec_errs, _, _, _, visited_idxes, decoded_all, _, _, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, rec_errs, _, _, _, visited_idxes, decoded_all, _, _, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True})

                # sampled_target_idx
                stix = [np.random.choice(valid_out_elevations, size=batch_size),
                        np.random.choice(  valid_out_azimuths, size=batch_size)]
                target_views = state.views_prepro_shifted[range(batch_size), stix[0], stix[1]] # (batch_size, C, 32, 32)
                belief_viewgrid = decoded_all[-1].cpu().data.numpy() # (batch_size, N, M, C, 32, 32)
                target_views_unsq = target_views[:, np.newaxis, np.newaxis, :, :, :]
                per_view_scores = -((belief_viewgrid - target_views_unsq)**2)
                per_view_scores = per_view_scores.reshape(batch_size, opts.N, opts.M, -1).mean(axis=3)
                # (batch_size, N, M)
                predicted_target_locs = np.unravel_index(np.argmax(per_view_scores.reshape(batch_size, -1), axis=1), (opts.N, opts.M))

                predicted_target_angles = []
                gt_target_angles = []
                for b in range(batch_size):
                    e, a = predicted_target_locs[0][b], predicted_target_locs[1][b]
                    predicted_target_angles.append(opts.idx_to_angles[(e, a)])
                    e, a = stix[0][b], stix[1][b]
                    gt_target_angles.append(opts.idx_to_angles[(e, a)])
                predicted_target_angles = np.array(predicted_target_angles)
                gt_target_angles = np.array(gt_target_angles)
                azim_err = norm_angles(predicted_target_angles[:, 1] - gt_target_angles[:, 1])
                elev_err = norm_angles(predicted_target_angles[:, 0] - gt_target_angles[:, 0])
                azim_errors_all.append(np.abs(azim_err))
                elev_errors_all.append(np.abs(elev_err))

    agent.policy.train()
    azim_errors_all = np.concatenate(azim_errors_all, axis=0)
    elev_errors_all = np.concatenate(elev_errors_all, axis=0)
    avg_azim_err = math.degrees(np.mean(azim_errors_all))
    avg_elev_err = math.degrees(np.mean(elev_errors_all))

    return avg_azim_err, avg_elev_err

def evaluate_pose(loader, agent, split, opts):
    """
    Evaluation of the Pose agent itself
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    azim_errors_all = []
    elev_errors_all = []
    decoded_images = []
    while not depleted:
        # ---- Sample batch of data ----
        if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
            pano, pano_maps, depleted = loader.next_batch(split)
            pano_rewards = None
        elif opts.expert_rewards:
            pano, pano_rewards, depleted = loader.next_batch(split)
            pano_maps = None
        else:
            pano, depleted = loader.next_batch(split)
            pano_rewards, pano_maps = None, None

        # Initial setup for evaluating over a grid of views
        batch_size = pano.shape[0]
	# Compute the performance with the initial state
        # starting at fixed grid locations
        if opts.start_view == 0:
            # Randomly sample one location from grid
            elevations = [random.randint(0, opts.N-1)]
            azimuths = [random.randint(0, opts.M-1)]
        elif opts.start_view == 1:
            # Sample only the center location from grid
            elevations = [opts.N//2]
            azimuths = [opts.M//2-1]
        elif opts.start_view == 2:
            elevations = range(0, opts.N, 2)
            azimuths = range(0, opts.M, 2)

        valid_out_elevations = np.array(range(0, opts.N))
        valid_out_azimuths = np.array(range(0, opts.M))
        elev_to_vis = elevations[random.randint(0, len(elevations)-1)]
        azim_to_vis = azimuths[random.randint(0, len(azimuths)-1)]

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'demo_sidekick' or opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
                    _, _, _, _,  _, visited_idxes, decoded_all, _, _, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)
                else:
                    _, _, _, _,  _, visited_idxes, decoded_all, _, _, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy, 'memorize_views': True}, pano_maps=pano_maps, opts=opts)

                # sampled_target_idx
                stix = [np.random.choice(valid_out_elevations, size=batch_size),
                        np.random.choice(  valid_out_azimuths, size=batch_size)]
                target_views = state.views_prepro_shifted[range(batch_size), stix[0], stix[1]] # (batch_size, C, 32, 32)
                belief_viewgrid = decoded_all[-1].cpu().data.numpy() # (batch_size, N, M, C, 32, 32)
                target_views_unsq = target_views[:, np.newaxis, np.newaxis, :, :, :]
                per_view_scores = -((belief_viewgrid - target_views_unsq)**2)
                per_view_scores = per_view_scores.reshape(batch_size, opts.N, opts.M, -1).mean(axis=3)
                # (batch_size, N, M)
                predicted_target_locs = np.unravel_index(np.argmax(per_view_scores.reshape(batch_size, -1), axis=1), (opts.N, opts.M))

                predicted_target_angles = []
                gt_target_angles = []
                for b in range(batch_size):
                    e, a = predicted_target_locs[0][b], predicted_target_locs[1][b]
                    predicted_target_angles.append(opts.idx_to_angles[(e, a)])
                    e, a = stix[0][b], stix[1][b]
                    gt_target_angles.append(opts.idx_to_angles[(e, a)])
                predicted_target_angles = np.array(predicted_target_angles)
                gt_target_angles = np.array(gt_target_angles)
                azim_err = norm_angles(predicted_target_angles[:, 1] - gt_target_angles[:, 1])
                elev_err = norm_angles(predicted_target_angles[:, 0] - gt_target_angles[:, 0])
                azim_errors_all.append(np.abs(azim_err))
                elev_errors_all.append(np.abs(elev_err))

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

                        # Fill in blue margin for the GT view whose pose needs to be estimated
                        for idx in range(batch_size):
                            elev_idx = stix[0][idx]
                            # Assumes that the azimuth is unknown initially
                            azim_idx = (start_idx[idx][1] + stix[1][idx]) % (opts.M)
                            true_state[idx, elev_idx, azim_idx, :, 0:3, :] = 0
                            true_state[idx, elev_idx, azim_idx, :, -3:, :] = 0
                            true_state[idx, elev_idx, azim_idx, :, :, 0:3] = 0
                            true_state[idx, elev_idx, azim_idx, :, :, -3:] = 0
                            true_state[idx, elev_idx, azim_idx, 2, 0:3, :] = 255
                            true_state[idx, elev_idx, azim_idx, 2, -3:, :] = 255
                            true_state[idx, elev_idx, azim_idx, 2, :, 0:3] = 255
                            true_state[idx, elev_idx, azim_idx, 2, :, -3:] = 255

                        # Fill in green margin for the view corresponding to the predicted pose
                        for idx in range(batch_size):
                            elev_idx = predicted_target_locs[0][idx]
                            # Assumes that the azimuth is unknown initially
                            azim_idx = (start_idx[idx][1] + predicted_target_locs[1][idx]) % (opts.M)
                            true_state[idx, elev_idx, azim_idx, :, 0:3, :] = 0
                            true_state[idx, elev_idx, azim_idx, :, -3:, :] = 0
                            true_state[idx, elev_idx, azim_idx, :, :, 0:3] = 0
                            true_state[idx, elev_idx, azim_idx, :, :, -3:] = 0
                            true_state[idx, elev_idx, azim_idx, 1, 0:3, :] = 255
                            true_state[idx, elev_idx, azim_idx, 1, -3:, :] = 255
                            true_state[idx, elev_idx, azim_idx, 1, :, 0:3] = 255
                            true_state[idx, elev_idx, azim_idx, 1, :, -3:] = 255


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

    agent.policy.train()
    azim_errors_all = np.concatenate(azim_errors_all, axis=0)
    elev_errors_all = np.concatenate(elev_errors_all, axis=0)
    avg_azim_err = math.degrees(np.mean(azim_errors_all))
    avg_elev_err = math.degrees(np.mean(elev_errors_all))

    return avg_azim_err, avg_elev_err, decoded_images
