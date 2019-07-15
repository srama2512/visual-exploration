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

def load_module(agent, opts):
    """
    Given the agent, load a pre-trained model and other setup based on the
    training_setting
    """
    # ---- Load the pre-trained model ----
    load_state = torch.load(opts.load_model)
    # strict=False ensures that only the modules common to loaded_dict and agent.policy's state_dict are loaded. 
    # Could potentially lead to errors being masked. Tread carefully! 
    agent.policy.load_state_dict(load_state['state_dict'], strict=False) 

def evaluate(loader, agent, split, opts):
    """
    Evaluation function - evaluates the agent over fixed grid locations as
    starting points and returns the overall average reconstruction error.
    """
    # ---- Initial setup ----
    depleted = False
    agent.policy.eval()
    for t in range(opts.T):
        agent.classifiers[t].eval()

    overall_err = 0
    overall_count = 0
    predicted_labs = []
    true_labs = []
    predicted_activations = []
    visited_idxes = []
    entropy_all = []
    if hasattr(opts, 'compute_all_times') and opts.compute_all_times:
        predicted_labs_all_times = []

    while not depleted:
        # ---- Sample batch of data ----
        if opts.actorType == 'saved_trajectories' or opts.actorType == 'peek_saliency':
            pano, labs, pano_maps, depleted = loader.next_batch(split)
            pano_rewards = None
        elif opts.expert_rewards:
            pano, labs, pano_rewards, depleted = loader.next_batch(split)
            pano_maps = None
        else:
            pano, labs, depleted = loader.next_batch(split)
            pano_rewards = None
            pano_maps = None

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
            # Sampling at uniform distances
            elevations = range(0, opts.N, 2)
            azimuths = range(0, opts.M, 2)

        vis_choice = (random.choice(elevations), random.choice(azimuths))
        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, labs, pano_rewards, start_idx, opts)
                _, _, _, entropy, classifier_activations_all, _, _, visited_idxes_curr, _, = agent.gather_trajectory(state, eval_opts={'greedy': opts.greedy}, pano_maps=pano_maps)
                if vis_choice[0] == i and vis_choice[1] == j:
                    predicted_activations.append([cacts.data.cpu() for cacts in classifier_activations_all])
                    visited_idxes.append(visited_idxes_curr)
                
                if hasattr(opts, 'compute_all_times') and opts.compute_all_times:
                    predicted_labs_curr = []
                    for k in range(len(classifier_activations_all)):
                        predicted_labs_curr.append(np.argmax(classifier_activations_all[k].data.cpu().numpy(), axis=1))
                    predicted_labs_all_times.append(predicted_labs_curr)

                if hasattr(opts, 'average_over_time') and opts.average_over_time is True:
                    mean_activation = 0
                    for k in range(len(classifier_activations_all)):
                        mean_activation += classifier_activations_all[k].data.cpu().numpy()
                    mean_activation /= len(classifier_activations_all)
                    predicted_labs.append(np.argmax(mean_activation, axis=1))
                else:
                    predicted_labs.append(np.argmax(classifier_activations_all[-1].data.cpu().numpy(), axis=1))
                true_labs.append(labs)
                if entropy[0] is not None:
                    entropy_all.append(torch.stack(entropy, dim=1))

    if len(entropy_all) > 0:
        entropy_all = torch.cat(entropy_all, dim=0)
        #print('Overall entropy: {}'.format(entropy_all.mean()))
    true_labs = np.concatenate(true_labs, axis=0)
    predicted_labs = np.concatenate(predicted_labs, axis=0)
    accuracy = accuracy_score(true_labs, predicted_labs)

    if hasattr(opts, 'compute_all_times') and opts.compute_all_times:
        accuracy_all_times = []
        for t in range(opts.T):
            predicted_labs_t = np.concatenate([_x[t] for _x in predicted_labs_all_times], axis=0)
            accuracy_t = accuracy_score(true_labs, predicted_labs_t)
            accuracy_all_times.append(accuracy_t)

    agent.policy.train()
    for t in range(opts.T):
        agent.classifiers[t].train()

    if hasattr(opts, 'compute_all_times') and opts.compute_all_times:
        return accuracy, accuracy_all_times, predicted_activations, visited_idxes
    else:
        return accuracy, predicted_activations, visited_idxes 
