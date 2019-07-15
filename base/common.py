import cv2
import random
import numpy as np

import torch
import torch.nn as nn

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def normalize_views(pano, mean, std):
    """
    Preprocesses the input views by subtracting the mean and dividing by the standard deviation
    pano: BxNxMxCxHxW numpy array
    """
    pano_float = pano.astype(np.float32)

    for c in range(len(mean)):
        pano_float[:, :, :, c, :, :] -= mean[c]
    return pano_float/255.0 # Scale pixel values from [0, 255] to [0, 1] range

def set_random_seeds(seed):
    """
    Sets the random seeds for numpy, python, pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_starts(N, M, batch_size, option):
    """
    Given the number of elevations(N), azimuths(M), batch size and the option (different types of starts),
    this function returns the start indices for the batch.
    start_idx: list of [start_elev, start_azim] for each panorama in the batch
    """
    if option == 0:
        start_idx = [[random.randint(0, N-1), random.randint(0, M-1)] for i in range(batch_size)]
    else:
        start_idx = [[N//2, M//2-1] for i in range(batch_size)]

    return start_idx

def utility_function(utility_matrix, selected_views, threshold):
    """
    Evaluates the quality of the selected views based on the utility_matrix
    utility_matrix : NxMxNxM array
    selected_views : list of (i, j) pairs indicating selected views
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    total_utility_map = np.zeros((N, M))

    for view in selected_views:
        total_utility_map += utility_matrix[view[0], view[1]]

    total_utility_map = np.minimum(total_utility_map, threshold)
    return total_utility_map.sum()

def utility_function_unique(utility_matrix, selected_views, threshold):
    """
    Evaluates the quality of the selected views based on the utility_matrix.
    This selects only uniques views for computation, to ensure that the
    same view does get selected multiple times.

    utility_matrix : NxMxNxM array
    selected_views : list of (i, j) pairs indicating selected views
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    total_utility_map = np.zeros((N, M))

    selected_views_set = set()
    for view in selected_views:
        selected_views_set.add((view[0], view[1]))

    for view in selected_views_set:
        total_utility_map += utility_matrix[view[0], view[1]]

    total_utility_map = np.minimum(total_utility_map, threshold)
    return total_utility_map.sum()

def get_submodular_views(utility_matrix, num_views):
    """
    Uses greedy maximization of submodular utility function to get close to optimal set of views
    utility_matrix : NxMxNxM array
    num_views      : number of views to select
    """
    M = utility_matrix.shape[1]
    N = utility_matrix.shape[0]
    sel_views = []

    total_utility = 0
    for n in range(num_views):
        max_idx = [0, 0]
        max_utility_gain = 0
        for i in range(N):
            for j in range(M):
                curr_utility_gain = utility_function(utility_matrix, sel_views + [[i, j]], 1) - total_utility
                if curr_utility_gain >= max_utility_gain:
                    max_utility_gain = curr_utility_gain
                    max_idx = [i, j]
        sel_views.append(max_idx)
        total_utility += max_utility_gain

    return sel_views, total_utility

def get_expert_trajectories(state, pano_maps_orig, selected_views, opts):
    """
    Get greedy trajectories based on utility for each panorama in batch
    opts must contain:
    T, delta_M, delta_N, wrap_elevation, wrap_azimuth, N, M
    """
    pano_maps = np.copy(pano_maps_orig)
    batch_size = pano_maps.shape[0]
    # Note: Assuming atleast one view has been selected initially
    t_start = len(selected_views[0])-1 # What t to start from, if some views have already been selected
    # Access pattern: selected_views[batch_size][time_step]
    selected_actions = np.zeros((batch_size, opts.T-t_start-1), np.int32)  # Access pattern: selected_actions[batch_size][time_step]
    for i in range(batch_size):
        curr_utility = utility_function_unique(pano_maps[i], selected_views[i], 1)
        # Given the first view, select T-1 more views
        t = t_start
        while t < opts.T-1:
            curr_pos = selected_views[i][t]
            max_gain = 0
            max_delta = None
            max_pos = None
            for delta_ele in range(-(opts.delta_N//2), opts.delta_N//2 + 1):
                for delta_azi in range(-(opts.delta_M//2), opts.delta_M//2 + 1):
                    if opts.wrap_elevation:
                        new_ele = (curr_pos[0] + delta_ele)%opts.N
                    else:
                        new_ele = max(min(curr_pos[0] + delta_ele, opts.N-1), 0)

                    if opts.wrap_azimuth:
                        new_azi = (curr_pos[1] + delta_azi)%opts.M
                    else:
                        new_azi = max(min(curr_pos[1] + delta_azi, opts.M-1), 0)

                    new_pos = [new_ele, new_azi]
                    curr_gain = utility_function_unique(pano_maps[i], selected_views[i] + [new_pos], 1) - curr_utility
                    if curr_gain >= max_gain:
                        max_gain = curr_gain
                        max_delta = (delta_ele, delta_azi)
                        max_pos = new_pos

            curr_utility += max_gain
            selected_views[i].append(max_pos)
            selected_actions[i][t-t_start] = state.delta_to_act[max_delta]
            t += 1

    return selected_views, selected_actions

class View(nn.Module):

    def __init__(self, *shape):
        # shape is a list
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

def peek_saliency_action(saliency_scores, pos, delta_M, delta_N, delta_to_act):
    # saliency_scores: BxNxM numpy array
    # pos: B list of 2 lists - current position
    batch_size = saliency_scores.shape[0]
    N, M = saliency_scores.shape[1:]
    actions = []
    for i in range(batch_size):
        max_score = -1000
        max_score_idx = (0, 0)
        # Elevation cannot be wrapped around
        s_elev = max((pos[i][0] - (delta_N//2)), 0)
        e_elev = min((pos[i][0] + (delta_N//2)), N-1)
        # Assumes that azimuth can be wrapped around
        s_azim = (pos[i][1] - (delta_M//2)) % M
        e_azim = (pos[i][1] + (delta_M//2)) % M
        # Scores in regions of interest
        region_scores = saliency_scores[i, s_elev:e_elev, s_azim:e_azim]
        max_score_idx = np.unravel_index(np.max(region_scores, axis=None), region_scores.shape)
        #if i == 0:
        #    print('max_score: {}'.format(max_score))
        #    print('max_score_idx: {}'.format(max_score_idx))
        actions.append(delta_to_act[max_score_idx])

    return actions

def initialize_sequential(var_sequential, init_method):
    """
    Given a sequential module (var_sequential) and an initialization method 
    (init_method), this initializes var_sequential using init_method

    Note: The layers returned are different from the one inputted. 
    Not sure if this affects anything.
    """
    var_list = []
    for i in range(len(var_sequential)):
        var_list.append(init_method(var_sequential[i]))

    return nn.Sequential(*var_list)

def iunf(input_layer, initunf=0.1):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.uniform(getattr(input_layer, 'weight_ih_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'weight_hh_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'bias_ih_l%d'%(i)), -initunf, initunf)
            nn.init.uniform(getattr(input_layer, 'bias_hh_l%d'%(i)), -initunf, initunf)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.uniform(input_layer.weight, -initunf, initunf);
        if hasattr(input_layer, 'bias'):
            nn.init.uniform(input_layer.bias, -initunf, initunf);
    return input_layer

def ixvr(input_layer, bias_val=0.01):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.xavier_normal(getattr(input_layer, 'weight_ih_l%d'%(i)))
            nn.init.xavier_normal(getattr(input_layer, 'weight_hh_l%d'%(i)))
            nn.init.constant(getattr(input_layer, 'bias_ih_l%d'%(i)), bias_val)
            nn.init.constant(getattr(input_layer, 'bias_hh_l%d'%(i)), bias_val)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.xavier_normal(input_layer.weight);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, bias_val);
    return input_layer

def inrml(input_layer, mean=0, std=0.001):
    # If the layer is an LSTM
    if str(type(input_layer)) == "<class 'torch.nn.modules.rnn.LSTM'>":
        for i in range(input_layer.num_layers):
            nn.init.normal(getattr(input_layer, 'weight_ih_l%d'%(i)), mean, std)
            nn.init.normal(getattr(input_layer, 'weight_hh_l%d'%(i)), mean, std)
            nn.init.constant(getattr(input_layer, 'bias_ih_l%d'%(i)), 0.01)
            nn.init.constant(getattr(input_layer, 'bias_hh_l%d'%(i)), 0.01)
    # For all other layers except batch norm
    elif not (str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm2d'>" or str(type(input_layer)) == "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>"):
        if hasattr(input_layer, 'weight'):
            nn.init.normal(input_layer.weight, mean, std);
        if hasattr(input_layer, 'bias'):
            nn.init.constant(input_layer.bias, 0.01);
    return input_layer

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
            int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
            int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
