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
from base.common import *
from sklearn.metrics import accuracy_score

def evaluate(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.eval()
    overall_err = 0
    overall_count = 0
    err_values = []
    gt_labs = []
    pred_labs = []
    vis_images = []
    while not depleted:
        # ---- Sample batch of data ----
        if opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
            pano, labs, pano_maps, depleted = loader.next_batch(split)
            pano_rewards = None
        else:
            pano, labs, depleted = loader.next_batch(split)
            pano_rewards = None
            pano_maps = None

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

        elev_to_vis = elevations[random.randint(0, len(elevations)-1)]
        azim_to_vis = azimuths[random.randint(0, len(azimuths)-1)]
        labs = Variable(torch.Tensor(labs))
        if opts.iscuda:
            labs = labs.cuda()

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, labs, pano_rewards, start_idx, opts)
                # Enable view memorization for testing by default
                if opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
                    _, pred_errs, _, _,  _, visited_idxes, predictions_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy}, pano_maps=pano_maps, opts=opts)
                else:
                    _, pred_errs, _, _,  _, visited_idxes, predictions_all, _ = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy})

                predictions_all = [pall.data.cpu() for pall in predictions_all]
                gt_labs.append(state.labs.cpu().data)
                pred_labs.append(predictions_all)

                # For some random initial state, print the decoded images at all time steps
                if i == elev_to_vis and j == azim_to_vis:
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

                    true_state = true_state.transpose((0, 3, 1, 4, 2, 5)).reshape(batch_size, 3, opts.N*32, opts.M*32)
                    true_state_final = []
                    # Draw arrows representing the actions on the true_state
                    for idx in range(batch_size):
                        true_state_curr = true_state[idx].transpose(1, 2, 0).copy()
                        for jdx in range(1, len(visited_idxes)):
                            elev_curr = visited_idxes[jdx][idx][0]
                            azim_curr = visited_idxes[jdx][idx][1]
                            elev_prev = visited_idxes[jdx-1][idx][0]
                            azim_prev = visited_idxes[jdx-1][idx][1]
                            arrow_start = (azim_prev * 32 + 16, elev_prev * 32 + 16)
                            arrow_end   = (azim_curr * 32 + 16, elev_curr * 32 + 16)
                            draw_arrow(true_state_curr, arrow_start, arrow_end, (255, 0, 0))
                        true_state_curr = np.pad(true_state_curr, ((60, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                        curr_label = state.labs[idx].data[0]
                        pred_label = predictions_all[-1][idx]
                        cv2.putText(true_state_curr, 'GT: {:.3f}, Pred: {:.3f}'.format(curr_label, pred_label), (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

                        true_state_final.append(true_state_curr.transpose(2, 0, 1))
                        #true_state[idx] = true_state_curr.transpose(2, 0, 1)
                    true_state = np.stack(true_state_final, axis=0)

                    vis_images.append(torch.Tensor(true_state/255.0))
                 
                # Add error from the last step
                curr_err += pred_errs[-1].data.sum()
                curr_count += 1 # Count for the views
                curr_err_batch += pred_errs[-1].data.cpu().numpy()

        curr_err /= curr_count
        curr_err_batch /= curr_count
        for i in range(curr_err_batch.shape[0]):
            err_values.append(float(curr_err_batch[i]))
        overall_err += curr_err
        overall_count += batch_size
    
    err_values = np.array(err_values)
    mean_err = float(np.mean(err_values))

    agent.train()

    pred_labs_across_time = [torch.cat([p[t] for p in pred_labs], dim=0) for t in range(opts.T)]
    gt_labs = torch.cat(gt_labs, dim=0)
    scores_across_time = []
    for t in range(opts.T):
        score = torch.mean(((gt_labs - pred_labs_across_time[t])/(gt_labs + 1e-8))**2)
        scores_across_time.append(math.sqrt(score))

    return mean_err, scores_across_time, vis_images
