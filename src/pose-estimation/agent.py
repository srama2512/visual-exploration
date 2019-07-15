from torch.autograd import Variable
from policy import Policy
from utils import *
from envs import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import itertools
import random
import torch
import copy
import pdb

from base.base_agent import BaseAgent
from base.common import *

class Agent(BaseAgent):
    def __init__(self, opts, mode='train'):
        super(Agent, self).__init__(opts, mode=mode)
        # Whether to memorize views or not
        self.memorize_views = opts.memorize_views
        self.idx_to_angles = opts.idx_to_angles
        # ---- Create the optimizer ----
        if self.mode == 'train':
            # ---- Load pre-trained weights ----
            loaded_state_dict = torch.load(opts.load_model)['state_dict']
            filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if ('act.' not in k and 'critic.' not in k)}
            state_dict = self.policy.state_dict()
            state_dict.update(filtered_state_dict)
            self.policy.load_state_dict(state_dict)
            # ---- Freeze pre-trained layers in the policy ----
            for params in itertools.chain(self.policy.sense_im.parameters(), self.policy.sense_pro.parameters(),
                                          self.policy.fuse.parameters(), self.policy.aggregate.parameters(),
                                          self.policy.decode.parameters()):
                params.requires_grad = False
            self.create_optimizer(opts.lr, opts.weight_decay)

    def gather_trajectory(self, state_object, eval_opts=None, pano_maps=None, opts=None):
        """
        gather_trajectory gets an observation, updates it's belief of the state, decodes the
        panorama and takes the next action. This is done repeatedly for T time steps.

        Note:
        eval_opts are provided only during testing
        pano_maps, opts are provided only when the actor is demo_sidekick
        """
        # This makes sure that the original pano_maps is not modified
        if pano_maps is not None and type(pano_maps) is not dict:
            pano_maps = np.copy(pano_maps)

        # ---- Setup variables to store trajectory information ----
        rewards = []
        log_probs = []
        elev_errors_all = []
        azim_errors_all = []
        entropies = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        decoded_all = []
        values = []
        actions_taken = torch.zeros(batch_size, self.T-1)
        valid_out_elevations = np.array(range(0, self.N))
        valid_out_azimuths = np.array(range(0, self.M))

        if (self.baselineType == 'critic' and self.critic_full_obs) or (self.actorType == 'actor' and self.act_full_obs):
            pano_input = torch.Tensor(state_object.views_prepro)

        if self.actorType == 'demo_sidekick':
            start_idx = state_object.start_idx
            # ---- Get the expert planned trajectories ----
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)
        elif self.actorType == 'saved_trajectories':
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0)
        elif self.actorType == 'const_action':
            target_actions = torch.LongTensor(batch_size, 1).fill_(self.policy.const_act)

        # sampled_target_idx for pose estimation
        stix = [np.random.choice(valid_out_elevations, size=batch_size), 
                np.random.choice(  valid_out_azimuths, size=batch_size)]
        target_views = state_object.views_prepro_shifted[range(batch_size), stix[0], stix[1]] # (batch_size, C, 32, 32)
        target_views_unsq = target_views[:, np.newaxis, np.newaxis, :, :, :]
        for t in range(self.T):
            # ---- Observe the panorama ----
            im, pro = state_object.get_view()
            im, pro = torch.Tensor(im), torch.Tensor(pro)
            # Keep track of visited locations
            visited_idxes.append(state_object.idx)
            # ---- Policy forward pass ----
            policy_input = {'im': im, 'pro': pro}
            # If critic or act have full observability, then elev, azim and time must be included in policy_input
            # along with the batch of panoramas
            if (self.baselineType == 'critic' and self.critic_full_obs) or (self.actorType == 'actor' and self.act_full_obs):
                policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
                policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
                policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])
                policy_input['pano'] = pano_input 
            else:
                if self.actOnElev:
                    policy_input['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
                if self.actOnAzim:
                    policy_input['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
                if self.actOnTime:
                    policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for var in policy_input:
                    policy_input[var] = policy_input[var].cuda()

            for var in policy_input:
                policy_input[var] = Variable(policy_input[var])

            # Note: decoded and hidden correspond to the previous transition
            # probs and value correspond to the new transition, where the value
            # and action probabilities of the current state are estimated for PG update.
            probs, hidden, decoded, value = self.policy.forward(policy_input, hidden)

            # ---- Memorize views ----
            # Memorize while testing only
            if self.memorize_views or (eval_opts is not None and eval_opts['memorize_views'] == True):
                for i in range(len(visited_idxes)):
                    for bno in range(batch_size):
                        # Shifting it to use appropriately for the decoded images and views_prepro_shifted
                        elev_start = state_object.start_idx[bno][0]
                        azim_start = state_object.start_idx[bno][1]
                        elev_curr = visited_idxes[i][bno][0]
                        azim_curr = visited_idxes[i][bno][1]

                        if not self.knownElev:
                            elev_visited = elev_curr - elev_start
                        else:
                            elev_visited = elev_curr

                        if not self.knownAzim:
                            azim_visited = azim_curr - azim_start
                        else:
                            azim_visited = azim_curr

                        view_copied = state_object.views_prepro_shifted[bno][elev_visited][azim_visited]

                        view_copied = Variable(torch.Tensor(view_copied))
                        if self.iscuda:
                            view_copied = view_copied.cuda()
                        decoded[bno, elev_visited, azim_visited] = view_copied

            decoded_all.append(decoded)

            # ---- Compute pose loss (corresponding to the previous transition)----
            if t == self.T-1:
                belief_viewgrid = decoded.cpu().data.numpy() # (batch_size, N, M, C, 32, 32)
                per_view_scores = -((belief_viewgrid - target_views_unsq)**2)
                per_view_scores = per_view_scores.reshape(batch_size, self.N, self.M, -1).mean(axis=3)
                predicted_target_locs = np.unravel_index(np.argmax(per_view_scores.reshape(batch_size, -1), axis=1), (self.N, self.M))

                predicted_target_angles = []
                gt_target_angles = []
                for b in range(batch_size):
                    e, a = predicted_target_locs[0][b], predicted_target_locs[1][b]
                    predicted_target_angles.append(self.idx_to_angles[(e, a)])
                    e, a = stix[0][b], stix[1][b]
                    gt_target_angles.append(self.idx_to_angles[(e, a)])
                predicted_target_angles = np.array(predicted_target_angles)
                gt_target_angles = np.array(gt_target_angles)
                azim_err = np.abs(norm_angles(predicted_target_angles[:, 1] - gt_target_angles[:, 1]))
                elev_err = np.abs(norm_angles(predicted_target_angles[:, 0] - gt_target_angles[:, 0]))
                azim_errors_all.append(np.degrees(azim_err))
                elev_errors_all.append(np.degrees(elev_err))
                pose_reward = -(azim_err + elev_err)

            # Pose reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = torch.Tensor(pose_reward) # Disconnects reward from future updates
                if self.iscuda:
                    reward = reward.cuda()
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            if t > 0:
                rewards[t-1] += reward

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate
            if t < self.T - 1:
                if self.actorType == 'actor': 
                    # Act based on the policy network
                    if eval_opts == None or eval_opts['greedy'] == False:
                        act = probs.multinomial(num_samples=1).data
                    else:
                        # This works only while evaluating, not while training
                        _, act = probs.max(dim=1)
                        act = act.data.view(-1, 1)
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing)
                    log_prob = (probs[range(act.size(0)), act[:, 0]]+1e-7).log()
                
                elif self.policy.actorType == 'random':
                    # Act randomly
                    act = torch.Tensor(np.random.randint(0, self.policy.A, size=(batch_size, 1)))
                    log_prob = None
                    entropy = None
                
                elif self.policy.actorType == 'greedyLookAhead':
                    # Accumulate scores for each batch for every action
                    act_scores_ = torch.ones(batch_size, self.policy.A).fill_(10000)
                    # For each action, compute the next state, perform the policy forward pass and obtain
                    # reconstruction error. 
                    for a_iter in range(self.policy.A):
                        state_object_ = copy.deepcopy(state_object) 
                        _ = state_object_.rotate(torch.ones(batch_size).fill_(a_iter).int())
                        im_, pro_ = state_object_.get_view()
                        im_, pro_ = torch.Tensor(im_), torch.Tensor(pro_)
                        policy_input_ = {'im': im_, 'pro': pro_}
                        # Assume no critic or proprioception, time inputs needed if greedyLookAhead is performed
                        if self.iscuda:
                            for var in policy_input_:
                                policy_input_[var] = policy_input_[var].cuda()

                        for var in policy_input_:
                            policy_input_[var] = Variable(policy_input_[var])
                        _, _, decoded_, _ = self.policy.forward(policy_input_, hidden)

                        rec_err_ = state_object_.loss_fn(decoded_, self.iscuda)
                        act_scores_[:, a_iter] = rec_err_.data.cpu()

                    _, act = torch.min(act_scores_, dim=1)
                    act = act.view(-1, 1)
                    log_prob = None
                    entropy = None

                elif self.policy.actorType == 'demo_sidekick':
                    act = torch.LongTensor(target_actions[:, t]).contiguous().view(-1, 1)
                    log_prob = None
                    entropy = None

                elif self.policy.actorType == 'saved_trajectories':
                    # Use target_actions
                    act = target_actions[:, t].contiguous().view(-1, 1)
                    log_prob = None
                    entropy = None

                elif self.policy.actorType == 'peek_saliency':
                    # Set the saliency scores for current positions to zero
                    pos = state_object.idx
                    for i in range(batch_size):
                        pano_maps[i, pos[i][0], pos[i][1]] = 0.0
                    target_actions = torch.Tensor(peek_saliency_action(pano_maps, state_object.idx, self.delta_M, self.delta_N, state_object.delta_to_act))
                    act = target_actions.contiguous().view(-1, 1)
                    log_prob = None
                    entropy = None

                elif self.policy.actorType == 'const_action':
                    act = target_actions.clone()
                    log_prob = None
                    entropy = None

                # ---- Rotate the view of the state and collect expert reward for this transition ----
                actions_taken[:, t] = act[:, 0]
                reward_expert = state_object.rotate(act[:, 0])
                reward_expert = torch.Tensor(reward_expert)
                if self.iscuda:
                    reward_expert = reward_expert.cuda()
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size

                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)

        mean_elev_error = np.concatenate(elev_errors_all).mean()
        mean_azim_error = np.concatenate(azim_errors_all).mean()

        return log_probs, rewards, entropies, decoded, values, visited_idxes, decoded_all, actions_taken, mean_elev_error, mean_azim_error

    def cuda(self):
        self.policy.cuda()

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def _create_policy(self, opts):
        self.policy = Policy(opts)

    def _get_parameters(self):
        # Train only the act and critic modules. Everything else is pre-trained and fixed.
        list_of_params = [{'params': self.policy.act.parameters()}]
        if hasattr(self.policy, 'critic'):
            list_of_params.append({'params': self.policy.critic.parameters()})
        return list_of_params
