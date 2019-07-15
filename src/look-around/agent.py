from torch.autograd import Variable
from policy import Policy
from utils import *
from envs import *

import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import copy
import pdb

from base.base_agent import BaseAgent
from base.common import *

class Agent(BaseAgent):
    """
    This agent implements the policy from Policy class and uses REINFORCE / Actor-Critic for policy improvement
    """
    def __init__(self, opts, mode='train'):
        super(Agent, self).__init__(opts, mode=mode)
        # Whether to memorize views or not
        self.memorize_views = opts.memorize_views

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
        rec_errs = []
        entropies = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        decoded_all = []
        values = []
        actions_taken = torch.zeros(batch_size, self.T-1)

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

        for t in range(self.T):
            # ---- Observe the panorama ----
            im, pro = state_object.get_view()
            im, pro = torch.Tensor(im), torch.Tensor(pro)
            # Keep track of visited locations
            visited_idxes.append(state_object.idx)

            # ---- Policy forward pass ----
            policy_input = {'im': im, 'pro': pro}
            # If critic or act have full observability, then elev, azim and time must
            # be included in policy_input along with the batch of panoramas
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

            # ---- Compute reconstruction loss (corresponding to the previous transition)----
            rec_err = state_object.loss_fn(decoded, self.iscuda)

            # Reconstruction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -rec_err.data # Disconnects reward from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            if t > 0:
                rewards[t-1] += reward

            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)

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

        return log_probs, rec_errs, rewards, entropies, decoded, values, visited_idxes, decoded_all, actions_taken

    def create_optimizer(self, lr, weight_decay, training_setting=0, fix_decode=False):
        # Can be used to create the optimizer
        parameters = self._get_parameters(training_setting=training_setting, fix_decode=fix_decode)
        self.optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    def cuda(self):
        self.policy.cuda()

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def _create_policy(self, opts):
        self.policy = Policy(opts)

    def _get_parameters(self, training_setting=0, fix_decode=False):
        # Refer main.py for training_setting
        if training_setting == 0 or training_setting == 2 or training_setting == 4:
            list_of_params = [{'params': self.policy.parameters()}]
        elif training_setting == 1 or training_setting == 3:
            list_of_params = [{'params': self.policy.aggregate.parameters()}]
            if not fix_decode: 
                list_of_params.append({'params': self.policy.decode.parameters()})
            if hasattr(self.policy, 'act'):
                list_of_params.append({'params': self.policy.act.parameters()})
                if self.act_full_obs:
                    list_of_params.append({'params': self.policy.act_fuse.parameters()})
                if hasattr(self.policy, 'critic'):
                    list_of_params.append({'params': self.policy.critic.parameters()})
                    if self.critic_full_obs:
                        list_of_params.append({'params': self.policy.critic_fuse.parameters()})
        return list_of_params

class AgentSupervised(Agent):
    """
    Uses imitation learning to learn the policy
    """
    def __init__(self, opts, mode='train'):
        # ---- NLL loss criterion for policy update ----
        self.criterion = nn.NLLLoss() # Defining this first since .cuda() is called in super()

        super(AgentSupervised, self).__init__(opts, mode=mode)

        # ---- Optimization settings ----
        self.supervised_scale = opts.supervised_scale
        self.trajectories_type = opts.trajectories_type
        self.T_sup = opts.T_sup

    def train_agent(self, state_object, pano_maps, opts):
        """
        train_agent gets a batch of panoramas and the optimal trajectories to take
        for these observations. It updates the policy based on the imitation learning.

        NOTE - this does not support full observability for critic, actor
        """
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        # ---- Get the expert planned trajectories ----
        # target_actions: B x T-1 array of integers between (0, self.policy.A-1)
        if opts.trajectories_type == 'utility_maps':
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)
        else:
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0).numpy()

        # ---- Setup variables to store trajectory information ----
        probs_all = []
        rec_errs = []
        hidden = None
        decoded_all = []
        visited_idxes = []
        # ---- Forward propagate trajectories through the policy ----
        for t in range(self.T):
            # Observe the panorama
            im, pro = state_object.get_view()
            im, pro = torch.Tensor(im), torch.Tensor(pro)
            # Keep track of visited locations
            visited_idxes.append(state_object.idx)
            # Policy forward pass
            policy_input = {'im': im, 'pro': pro}
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

            probs, hidden, decoded, value = self.policy.forward(policy_input, hidden)
            decoded_all.append(decoded)

            # ---- Compute reconstruction loss ----
            rec_err = state_object.loss_fn(decoded, self.iscuda)

            # ---- Sample action, except in the last step ----
            if t < self.T-1:
                act = target_actions[:, t]
                # ---- Rotate the view of the state ----
                _ = state_object.rotate(act)
                probs_all.append(probs)

            rec_errs.append(rec_err)

        # ---- Update the policy ----
        loss = 0
        for t in range(self.T):
            loss = loss + rec_errs[t].sum()/batch_size
            if t < self.T-1:
                targets = Variable(torch.LongTensor(target_actions[:, t]), requires_grad=False)
                if self.iscuda:
                    targets = targets.cuda()
                loss = loss + self.criterion((probs_all[t]+1e-8).log(), targets)*self.supervised_scale

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

        return rec_errs

    def train_agent_hybrid(self, state_object, pano_maps, opts):
        """
        train_agent_hybrid combines both imitation and reinforcement learning schemes.
        It uses the expert agent for first K time steps and then acts based on it's own policy.
        The rewards are used in PG updates for the actions based on the policy and
        supervised updates are used for the actions taken based on the expert agent.

        NOTE - this does not support full observability for critic, actor
        """
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        # ---- Get the expert planned trajectories ----
        if opts.trajectories_type == 'utility_maps':
            selected_views = []
            for i in range(batch_size):
                selected_views.append([start_idx[i]])
            selected_views, target_actions = get_expert_trajectories(state_object, pano_maps, selected_views, opts)
        else:
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0).numpy()

        # ---- Setup variables to store trajectory information ----
        probs_all = []
        log_probs_all = []
        rec_errs = []
        values = []
        hidden = None
        visited_idxes = []
        rewards = []
        entropies = []
        decoded_all = []

        T_sup = self.T_sup

        # ---- Run the hybrid trajectory collection ----
        for t in range(self.T):
            # ---- Observe the panorama ----
            # Assuming actorType == 'actor'
            im, pro = state_object.get_view()
            im, pro = torch.Tensor(im), torch.Tensor(pro)
            # Store the visited locations
            visited_idxes.append(state_object.idx)

            # ---- Create the policy input ----
            input_policy = {'im': im, 'pro': pro}

            if self.actOnElev:
                input_policy['elev'] = torch.Tensor([[state_object.idx[i][0]] for i in range(batch_size)])
            if self.actOnAzim:
                input_policy['azim'] = torch.Tensor([[state_object.idx[i][1]] for i in range(batch_size)])
            if self.actOnTime:
                input_policy['time'] = torch.Tensor([[t] for i in range(batch_size)])

            if self.iscuda:
                for x in input_policy:
                    input_policy[x] = input_policy[x].cuda()

            for x in input_policy:
                input_policy[x] = Variable(input_policy[x])

            probs, hidden, decoded, value = self.policy.forward(input_policy, hidden)
            decoded_all.append(decoded)

            # ---- Compute reconstruction loss (corresponding to previous transition) ----
            rec_err = state_object.loss_fn(decoded, self.iscuda)

            # Reconstruction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -rec_err.data # Disconnect rewards from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size

            if t > 0:
                rewards[t-1] += reward

            # There are self.T reconstruction errors as opposed to self.T-1 rewards
            rec_errs.append(rec_err)

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate
            if t < self.T - 1:
                if t < T_sup:
                    # Act according to the supervised agent
                    act = target_actions[:, t]
                    log_prob = None
                    entropy = None
                else:
                    # Act based on the policy network
                    act = probs.multinomial(num_samples=1).data[:, 0]
                    # Compute entropy
                    entropy = -(probs*((probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing_
                    log_prob = (probs[range(act.size(0)), act]+1e-7).log()

                # ---- Rotate the view of the state ----
                reward_expert = state_object.rotate(act)
                reward_expert = torch.Tensor(reward_expert)
                if self.iscuda:
                    reward_expert = reward_expert.cuda()

                if self.baselineType == 'average':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size

                # This is the intrinsic reward corresponding to the current action
                rewards.append(reward_expert*self.reward_scale_expert)
                log_probs_all.append(log_prob)
                entropies.append(entropy)
                probs_all.append(probs)
                values.append(value)

        # ---- Update the policy ----
        R = torch.zeros(batch_size) # Reward accumulator
        B = 0 # Baseline accumulator - used primarily for the average baseline case
        loss = 0
        if self.iscuda:
            R = R.cuda()

        # ---- Reconstruction error based loss computation ----
        for t in reversed(range(self.T)):
            loss = loss + rec_errs[t].sum()/batch_size

        # ---- REINFORCE / Actor-Critic / Supervised loss based on T-1 transitions
        for t in reversed(range(self.T-1)):
            if t < T_sup:
                targets = Variable(torch.LongTensor(target_actions[:, t]), requires_grad=False)
                if self.iscuda:
                    targets = targets.cuda()
                loss = loss + self.criterion((probs_all[t]+1e-7).log(), targets)*self.supervised_scale
            elif t < self.T-1:
                R = R + rewards[t] # A one sample MC estimate of Q[t]
                # Compute the advantage
                if self.baselineType == 'critic':
                    adv = R - values[t].data
                else:
                    # B - an estimate of V[t] when no critic is present. Equivalent to subtracting
                    # the average  rewards at each time.
                    if t == self.T-2:
                        B += self.R_avg
                    B += self.R_avg_expert * self.reward_scale_expert
                    adv = R - B

                loss_term_1 = -(log_probs_all[t]*self.reward_scale*Variable(adv, requires_grad=False)).sum()/batch_size # PG loss
                loss_term_2 = -self.lambda_entropy*entropies[t].sum()/batch_size # Entropy loss, maximize entropy
                # Critic prediction error
                if self.baselineType == 'critic':
                    loss_term_3 = self.critic_coeff*((Variable(R, requires_grad=False)-values[t])**2).sum()/batch_size
                else:
                    loss_term_3 = 0

                loss = loss + loss_term_1 + loss_term_2 + loss_term_3

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

        return rec_errs

    def cuda(self):
        self.policy = self.policy.cuda()
        self.criterion = self.criterion.cuda()
