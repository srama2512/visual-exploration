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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_rewards_sidekick(loader, agent, split, opts):
    """
    One View reward function - evaluates the MSE error for each view of the panorama and
    returns scores for each view.
    Outputs:
        true_images    : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        scores_matrices : list of BxNxM arrays, contains scores corresponding to each
                         location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    scores_matrices = []

    _normalize_ = lambda x, minv, maxv: (x - minv)/(maxv - minv + 1e-8)
    while not depleted:
        pano, depleted = loader.next_batch(split)

        curr_err = 0
        batch_size = pano.shape[0]
        # Compute the performance with the initial state
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)

        # Scores matrices are stored as BxNxM matrices with one value corresponding to one view.
        scores_matrix = np.zeros(pano.shape[0:3])

        for i in elevations:
            for j in azimuths:
                start_idx = [[i, j] for _ in range(batch_size)]
                state = State(pano, None, start_idx, opts)
                _, rec_errs, _, _,  _, _, _, _, _ = agent.gather_trajectory(state, {'greedy': opts.greedy, 'memorize_views': opts.memorize_views})
                rec_errs = rec_errs[-1].data.cpu()
                for k in range(batch_size):
                    reward = 1/(rec_errs[k]*1000)
                    scores_matrix[k, i, j] = reward

        # Rescale scores for each image in batch
        for i in range(batch_size):
            max_v = np.max(scores_matrix[i])
            min_v = np.min(scores_matrix[i])
            scores_matrix[i] = _normalize_(scores_matrix[i], min_v, max_v)

        true_images.append(pano)
        scores_matrices.append(scores_matrix)

    return true_images, scores_matrices

def get_rewards_random(loader, agent, split, opts):
    """
    This is a baseline mechanism where rewards are spread uniformly randomly throughout the
    different views.
    Outputs:
        true_images    : list of BxNxMxCx32x32 arrays (each element of list corresponds to one batch)
        scores_matrices : list of BxNxM arrays, contains scores corresponding to each
                         location in NxM panorama
    """
    depleted = False
    agent.policy.eval()
    true_images = []
    scores_matrices = []

    while not depleted:
        pano, depleted = loader.next_batch(split)

        curr_err = 0
        batch_size = pano.shape[0]
        # Compute the performance with the initial state 
        # starting at fixed grid locations
        elevations = range(0, opts.N)
        azimuths = range(0, opts.M)
        # Scores matrices are stored as BxNxM matrices with one value corresponding to one view. 
        scores_matrix = np.zeros(pano.shape[0:3])

        # Randomly sample nms_iters reward locations
        rnd_azi = np.random.randint(0, opts.M, (batch_size, opts.nms_iters))
        rnd_ele = np.random.randint(0, opts.N, (batch_size, opts.nms_iters))

        for i in range(batch_size):
            for j in range(opts.nms_iters):
                m_ele = rnd_ele[i, j]
                m_azi = rnd_azi[i, j]
                nbd = opts.nms_nbd
                for k1 in [value_1 % opts.N for value_1 in range(m_ele - nbd, m_ele + nbd + 1)]: 
                    for k2 in [value_2 % opts.M for value_2 in range(m_azi - nbd, m_azi + nbd + 1)]:
                        curr_value = 1.0 / (4.0 ** (max(abs(m_azi - k2), abs(m_ele - k1))))
                        scores_matrix[i, k1, k2] += curr_value

        scores_matrix = np.minimum(scores_matrix, 1)
        true_images.append(pano)
        scores_matrices.append(scores_matrix)

    return true_images, scores_matrices 

def find_max(matrix):
    # max idxes and values of a 2D matrix
    max_val = -100000000.0
    max_idx = (0, 0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if max_val <= matrix[i, j]:
                max_val = matrix[i, j]
                max_idx = (i, j)
    return max_val, max_idx

def greedy_nms(score_matrix_, nms_iters, nms_nbd, smoothen=True):
    """
    Takes in a BxNxM numpy array and performs NMS on
    each NxM matrix.
    Output: BxNxM numpy array 
    """
    score_matrix = np.copy(score_matrix_)
    shape = score_matrix.shape
    final_score_matrix = np.zeros_like(score_matrix)
    B, N, M = shape[:3]
    for i in range(B):
        iter_count = 0
        while iter_count < nms_iters:
            max_val, max_idx = find_max(score_matrix[i])
            # Adds +1 to the actual maxima location and 0.25 to the adjacent locations
            if smoothen:
                for j in [value_j%N for value_j in range(max_idx[0] - nms_nbd, max_idx[0] + nms_nbd + 1)]: 
                    for k in [value_k%M for value_k in range(max_idx[1] - nms_nbd, max_idx[1] + nms_nbd + 1)]:
                        final_score_matrix[i, j, k] += max_val/(4.0**(max(abs(max_idx[0] - j), abs(max_idx[1] - k))))
            else:
                final_score_matrix[i, max_idx[0], max_idx[1]] = max_val

            # Eliminate the maxima and neighbours for next iteration
            for j in [value_j%N for value_j in range(max_idx[0]-nms_nbd, max_idx[0]+nms_nbd+1)]: 
                for k in [value_k%M for value_k in range(max_idx[1]-nms_nbd, max_idx[1]+nms_nbd+1)]:
                    score_matrix[i, j, k] = 0
            
            iter_count += 1
        
    return final_score_matrix

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
    opts.h5_path_unseen = ''
    opts.utility_h5_path = ''
    opts.supervised_scale = 1e-2
    opts.reward_scale_expert = 1e-4
    opts.expert_trajectories = False
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

    loader = DataLoader(opts)
    agent = Agent(opts, mode='eval')
    agent.policy.load_state_dict(loaded_state['state_dict'])

    h5file = h5py.File(opts.save_path_h5, 'w')

    for split in ['train', 'val']:
        if opts.score_type == 3:
            true_images, scores_matrices = get_rewards_random(loader, agent, split, opts)
        else:
            true_images, scores_matrices = get_rewards_sidekick(loader, agent, split, opts)

        final_scores_matrices_nms = []
        for i in range(len(true_images)):
            scores_matrix = scores_matrices[i]
            if opts.score_type == 0:
                # NMS, but no smoothing
                scores_matrix_nms = greedy_nms(scores_matrix, opts.nms_iters, opts.nms_nbd, smoothen=False)
            elif opts.score_type == 1:
                # NMS + smoothing
                scores_matrix_nms = greedy_nms(scores_matrix, opts.nms_iters, opts.nms_nbd, smoothen=True)
            elif opts.score_type == 2 or opts.score_type == 3:
                # No NMS
                scores_matrix_nms = np.copy(scores_matrix)
            final_scores_matrices_nms.append(scores_matrix_nms)

        # introduce new axes for channel, height and width
        _matrix_to_img_ = lambda x, c, h, w: np.repeat(np.repeat(np.repeat(
                                            x[:, :, :, np.newaxis, np.newaxis, np.newaxis],
                                            c, axis=3), h, axis=4), w, axis=5)
        if split == 'val':
            images_count = 0
            # Iterate through the different batches
            for i in range(len(true_images)):
                B, N, M, C, H, W = true_images[i].shape
                true_images[i] = np.reshape(true_images[i], (B, N*M, C, H, W))/255.0
                scores_image_nms = _matrix_to_img_(final_scores_matrices_nms[i], C, H, W)
                scores_image_normal = _matrix_to_img_(scores_matrices[i], C, H, W)
                scores_image_nms = np.reshape(scores_image_nms, (B, N*M, C, H, W))
                scores_image_normal = np.reshape(scores_image_normal, (B, N*M, C, H, W))
                concatenated = torch.Tensor(np.concatenate([true_images[i], scores_image_normal, scores_image_nms], axis=1))
                for j in range(B):
                    x = vutils.make_grid(concatenated[j], padding=True, normalize=False, scale_each=False, nrow=opts.M)
                    images_count += 1

                    writer.add_image('Panorama #%5.3d'%(images_count), x, 0)

        scores_matrices = np.concatenate(scores_matrices, axis=0)
        final_scores_matrices_nms = np.concatenate(final_scores_matrices_nms, axis=0)
        h5file.create_dataset('%s/normal'%split, data=scores_matrices)
        h5file.create_dataset('%s/nms'%split, data=final_scores_matrices_nms)

    json.dump(vars(opts), open(opts.save_path_json, 'w'))
    writer.close()
    h5file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--batch_size', type=int, default=32)

    # Agent options
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=True)
    parser.add_argument('--save_path_vis', type=str, default='')
    parser.add_argument('--save_path_h5', type=str, default='rewards.h5')
    parser.add_argument('--save_path_json', type=str, default='rewards.json')

    # Environment options
    parser.add_argument('--nms_iters', type=int, default=4, help='number of maxima to select from nms')
    parser.add_argument('--nms_nbd', type=int, default=1, help='distance of neighbours to suppress')
    parser.add_argument('--score_type', type=int, default=0, help='[ 0 - perform nms only and extract maxima | \
                                                                     1 - perform nms and smoothen to spread out the reward \
                                                                         to neighbouring views \
                                                                     2 - do not perform nms \
                                                                     3 - uniformly spread out rewards (baseline)]')

    # Other options
    opts = parser.parse_args()

    main(opts)
