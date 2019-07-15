import os
import sys
import pdb

import json
import h5py
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

def get_utility_maps(loader, agent, split, opts):
    """
    get_utility_maps - computes the MSE error for using one view of panorama P and
    generating each view of P. This is used to generate a scoring map which defines 
    the utility (0-1) of picking a view V for reconstructing each of the views of P.  
    Outputs:
        true_images      : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        utility_images   : list of BxNxMxNxMxCx32x32 arrays, each MxNxCx32x32 image is a set of constant 
                           images containing utility map corresponding to each location in NxM panorama
        utility_matrices : list of BxNxMxNxM arrays, contains utility maps corresponding to each
                           location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    utility_images = []
    utility_matrices = []

    print('====> Processing {} split'.format(split))
    while not depleted:
        pano, depleted = loader.next_batch(split)

        curr_err = 0
        batch_size = pano.shape[0]

        N = pano.shape[1]
        M = pano.shape[2]
        C = pano.shape[3]
        H = 8
        W = 8

        # Compute the performance with the initial state
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        utility_image = np.zeros((batch_size, N, M, N, M, C, H, W))
        utility_matrix = np.zeros((batch_size, N, M, N, M))

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]

                state = State(pano, None, start_idx, opts)
                _, rec_errs, _, _,  decoded, _, _, _, _ = agent.gather_trajectory(state, {'greedy': opts.greedy, 'memorize_views': opts.memorize_views})

                rec_errs_per_view = (state.views_prepro_shifted - decoded.data.cpu().numpy())**2
                rec_errs_per_view = np.reshape(rec_errs_per_view, (batch_size, N, M, -1)).sum(axis=3)
                rec_errs = rec_errs[0].data.cpu().numpy()
                # Rotate the reconstruction errors by the starting view to get original orientation
                rec_errs_per_view = np.roll(rec_errs_per_view, j, axis=2)
                for k in range(batch_size):
                    utility_matrix[k, i, j] = 1/(rec_errs_per_view[k, :, :]*1000.0 + 1e-8)

        # Rescale utility by normalizing over the utility of taking any view @ a particular view
        #utility_matrix - (batch_size, N, M, N, M)
        max_v = np.max(np.max(utility_matrix, axis=2), axis=1)
        min_v = np.min(np.min(utility_matrix, axis=2), axis=1)
        # expanding the max and min to span over all views
        max_v = max_v[:, np.newaxis, np.newaxis, :, :]
        min_v = min_v[:, np.newaxis, np.newaxis, :, :]
        utility_matrix = (utility_matrix - min_v)/(max_v - min_v + 1e-8)

        if opts.threshold_maps:
            utility_matrix[utility_matrix > 0.5] = 1
            utility_matrix[utility_matrix <= 0.5] = 0

        # Zero pad the utility image for display
        utility_image = np.repeat(np.repeat(
                                  np.repeat(utility_matrix[:, :, :, :, :, np.newaxis,
                                                           np.newaxis, np.newaxis],
                                            repeats=C, axis=5), repeats=H, axis=6
                                  ), repeats=W, axis=7)

        true_images.append(pano)
        utility_images.append(utility_image)
        utility_matrices.append(utility_matrix)

    return true_images, utility_images, utility_matrices

def main(opts):
    loaded_state = torch.load(opts.load_model)

    opts.T = loaded_state['opts'].T
    opts.M = loaded_state['opts'].M
    opts.N = loaded_state['opts'].N
    opts.delta_M = loaded_state['opts'].delta_M
    opts.delta_N = loaded_state['opts'].delta_N
    opts.dataset = loaded_state['opts'].dataset
    opts.h5_path = loaded_state['opts'].h5_path
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
    opts.memorize_views = loaded_state['opts'].memorize_views
    opts.critic_full_obs = loaded_state['opts'].critic_full_obs
    opts.actOnElev = loaded_state['opts'].actOnElev
    opts.actOnElev = loaded_state['opts'].actOnElev
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N

    opts.seed = 123
    opts.init = 'xavier'
    opts.shuffle = False
    opts.reward_scale = 1.0
    opts.utility_h5_path = ''
    opts.supervised_scale = 1e-2
    opts.reward_scale_expert = 1e-4
    opts.expert_trajectories = False
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N
    # Set random seeds
    set_random_seeds(opts.seed)

    if opts.dataset == 0:
        if opts.mean_subtract:
            opts.mean = [119.16, 107.68, 95.12]
            opts.std = [61.88, 61.72, 67.24]
        else:
            opts.mean = [0, 0, 0]
            opts.std = [0, 0, 0]
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

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path_vis)

    print('====> Loading dataset')
    loader = DataLoader(opts)
    print('====> Creating agent and loading model')
    agent = Agent(opts, mode='eval')

    agent.policy.load_state_dict(loaded_state['state_dict'])

    h5file = h5py.File(opts.save_path_h5, 'w')

    all_splits = ['train', 'val', 'test']
    if opts.dataset == 1:
        all_splits.append('test_unseen')

    print('====> Processing data')
    for split in all_splits:
        # Compute utility maps for computing coverage
        true_images, utility_images, utility_matrices = get_utility_maps(loader, agent, split, opts)

        if split == 'val':
            images_count = 0
            # Iterate through the different batches
            for i in range(len(true_images)):
                shape = true_images[i].shape
                true_images[i] = np.reshape(true_images[i].transpose(0, 3, 1, 4, 2, 5), (shape[0], 1, shape[3], shape[1]*shape[4], shape[2]*shape[5]))/255.0
                utility_images_normal = np.reshape(utility_images[i].transpose(0, 1, 2, 5, 3, 6, 4, 7), (shape[0], opts.N*opts.M, opts.num_channels, opts.N*8, opts.M*8))
                for j in range(shape[0]):
                    x = vutils.make_grid(torch.Tensor(utility_images_normal[j]), padding=3, normalize=False, scale_each=False, nrow=opts.M)
                    images_count += 1
                    writer.add_image('Panorama #%5.3d utility'%(images_count), x, 0)
                    # Display the true panorama
                    x = vutils.make_grid(torch.Tensor(true_images[i][j]), padding=3, normalize=False, scale_each=False, nrow=1)
                    writer.add_image('Panorama #%5.3d image'%(images_count), x, 0)

        utility_matrices = np.concatenate(utility_matrices, axis=0)
        h5file.create_dataset('%s/utility_maps'%split, data=utility_matrices)

    json.dump(vars(opts), open(opts.save_path_json, 'w'))
    writer.close()
    h5file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=32)

    # Agent options
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--save_path_vis', type=str, default='')
    parser.add_argument('--save_path_h5', type=str, default='utility_maps.h5')
    parser.add_argument('--save_path_json', type=str, default='utility_maps.json')

    # Other options
    parser.add_argument('--threshold_maps', type=str2bool, default=False, help='Threshold the utility maps to get binary utilities')

    opts = parser.parse_args()
    opts.memorize_views = False
    opts.critic_full_obs = False
    opts.act_full_obs = False
    main(opts)
