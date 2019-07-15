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

from base.common import *

# Not using BaseAgent since the options vary quite a bit.
class Agent:
    """
    This agent implements the policy from Policy class and uses REINFORCE for policy improvement
    """
    def __init__(self, opts, mode='train'):
        # ---- Create the policy network ----
        self.policy = Policy(opts)
        # ---- Create the classifiers ----
        self.num_classes = opts.num_classes
        self.classifiers = []
        for t in range(opts.T):
            self.classifiers.append(nn.Sequential(
                                    nn.BatchNorm1d(256),
                                    nn.Linear(256, opts.num_classes))
                               )
            if opts.init == 'xavier':
                self.classifiers[t] = initialize_sequential(self.classifiers[t], ixvr)
            elif opts.init == 'normal': 
                self.classifiers[t] = initialize_sequential(self.classifiers[t], inrml)
            else:
                self.classifiers[t] = initialize_sequential(self.classifiers[t], iunf)

        # ---- Panorama operation settings ----
        self.T = opts.T
        self.M = opts.M
        self.N = opts.N
        self.F = opts.F
        self.delta_M = opts.delta_M
        self.delta_N = opts.delta_N
        self.act_to_delta = opts.act_to_delta
        # Whether elevation, azimuth or time are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime

        # ---- Optimization settings ----
        self.mode = mode
        self.baselineType = opts.baselineType
        self.iscuda = opts.iscuda
        self.optimizer_type = opts.optimizer_type
        if opts.iscuda:
            self.policy = self.policy.cuda()
            for t in range(opts.T):
                self.classifiers[t] = self.classifiers[t].cuda()
        if self.baselineType == 'average' and mode == 'train':
            self.R_avg_expert = 0
            self.avg_count_expert = 0
        self.frequent_class = opts.frequent_class
        # Use greedy rewards?
        self.rewards_greedy = opts.rewards_greedy

        # Scaling factors
        if self.mode == 'train':
            self.lambda_entropy = opts.lambda_entropy # Entropy term coefficient
            self.lambda_la = opts.lambda_la # look ahead term coefficient
            self.reward_scale = opts.reward_scale

        self.reward_scale_expert = opts.reward_scale_expert

        # ---- Create the optimizer ----
        if self.mode == 'train':
            self.create_optimizer(opts.lr, opts.weight_decay, opts.momentum)
            self.criterion_classification = nn.CrossEntropyLoss()

    def create_optimizer(self, lr, weight_decay, momentum):
        # Can be used to create the optimizer
        list_of_params = [{'params': self.policy.parameters()}]
        for t in range(self.T):
            list_of_params.append({'params': self.classifiers[t].parameters()})

        if self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(list_of_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(list_of_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError('Optimizer has to be adam (or) sgd!')

    def gather_trajectory(self, state_object, eval_opts=None, pano_maps=None):
        """
        gather_trajectory gets an observation, updates it's belief of the state, predicts the
        class and takes the next action. This is done repeatedly for T time steps.

        Note:
        eval_opts are provided only during testing, will not contribute
        to the training in any way
        """
        if (pano_maps is not None) and (type(pano_maps) is not dict):
            pano_maps = np.copy(pano_maps)

        # ---- Setup variables to store trajectory information ----
        reward_all = []
        baseline_all = []
        log_prob_act_all = []
        entropy_all = []
        lookahead_pred_all = []
        lookahead_gt_all = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        classifier_activations_all = []
        if self.T > 1:
            actions_taken = torch.zeros(batch_size, self.T-1)
        else:
            actions_taken = None

        if self.policy.actorType == 'saved_trajectories':
            target_actions = torch.cat([pano_maps[tuple(start_idx[i])][i, :].view(1, -1) for i in range(batch_size)], dim=0)
        elif self.policy.actorType == 'const_action':
            target_actions = torch.LongTensor(batch_size, 1).fill_(self.policy.const_act)

        for t in range(self.T):
            # ---- Observe the panorama ----
            im, delta, pro = state_object.get_view()
            im, delta, pro = torch.Tensor(im), torch.Tensor(delta), torch.Tensor(pro)
            # Keep track of visited locations
            visited_idxes.append(state_object.idx)

            # ---- Policy forward pass ----
            policy_input = {'im': im, 'delta': delta, 'pro': pro}

            if self.actOnTime:
                policy_input['time'] = torch.Tensor([[t] for i in range(batch_size)])
            if self.iscuda:
                for var in policy_input:
                    policy_input[var] = policy_input[var].cuda()
            for var in policy_input:
                policy_input[var] = Variable(policy_input[var])

            action_probs, hidden, lookahead_pred = self.policy.forward(policy_input, hidden)

            # ---- Compute classifier output
            if self.policy.rnn_type == 0 or self.policy.rnn_type == 2:
                if self.policy.normalize_hidden:
                    hidden_normalized = F.normalize(hidden.view(batch_size, -1), p=1, dim=1)
                else:
                    hidden_normalized = hidden.view(batch_size, -1)
                classifier_activations = self.classifiers[t](hidden_normalized)
            else:
                if self.policy.normalize_hidden:
                    hidden_normalized = F.normalize(hidden[0].view(batch_size, -1), p=1, dim=1)
                else:
                    hidden_normalized = hidden[0].view(batch_size, -1)
                classifier_activations = self.classifiers[t](hidden_normalized)

            classifier_activations_all.append(classifier_activations)

            if t > 0:
                # ---- Calculate the rewards obtained (for the previous action) ----
                # Note that rewards are not greedy by default.
                enable_rewards = False
                # If greedy, enable rewards at all time steps
                if self.rewards_greedy:
                    enable_rewards = True
                # else, enable rewards only at the last time step
                elif t == self.T-1:
                    enable_rewards = True

                if enable_rewards:
                    reward = state_object.reward_fn(classifier_activations.data.cpu().numpy())
                    reward = torch.Tensor(reward)
                    if self.iscuda:
                        reward = reward.cuda()
                    reward_all[t-1] += reward

                    # The baseline is the reward provided for the most common class at this point
                    baseline_class_pred = np.zeros((batch_size, self.num_classes))

                    baseline_class_pred[:, self.frequent_class] = 1
                    baseline = state_object.reward_fn(baseline_class_pred)
                    baseline = torch.Tensor(baseline)
                    if self.iscuda:
                        baseline = baseline.cuda()
                else:
                    baseline = torch.zeros(batch_size)
                    if self.iscuda:
                        baseline = baseline.cuda()

                baseline_all.append(baseline)
                # Add the lookahead ground truth
                if self.policy.rnn_type == 0 or self.policy.rnn_type == 2:
                    lookahead_gt_all.append(hidden.view(batch_size, -1))
                else:
                    lookahead_gt_all.append(hidden[0].view(batch_size, -1))

            # ---- Sample action ----
            # except for the last time step when only the selected view from previous step is used in aggregate 
            if t < self.T - 1:
                if self.policy.actorType == 'actor':
                    # Act based on the policy network
                    if eval_opts == None or eval_opts['greedy'] == False:
                        act = action_probs.multinomial(num_samples=1).data
                    else:
                        # This works only while evaluating, not while training
                        _, act = action_probs.max(dim=1)
                        act = act.data.view(-1, 1)
                    # Compute entropy
                    entropy = -(action_probs*((action_probs+1e-7).log())).sum(dim=1)
                    # Store log probabilities of selected actions (Advanced indexing)
                    log_prob_act = (action_probs[range(act.size(0)), act[:, 0]]+1e-7).log()

                elif self.policy.actorType == 'random':
                    # Act randomly
                    act = torch.Tensor(np.random.randint(0, self.policy.A, size=(batch_size, 1)))
                    log_prob_act = None
                    entropy = None

                elif self.policy.actorType == 'saved_trajectories':
                    # Use target_actions
                    act = target_actions[:, t].long().contiguous().view(-1, 1)
                    log_prob_act = None
                    entropy = None

                elif self.policy.actorType == 'peek_saliency':
                    # Set the saliency scroes for current positions to zero
                    pos = state_object.idx
                    for i in range(batch_size):
                        pano_maps[i][pos[i][0], pos[i][1]] = 0.0
                    target_actions = torch.Tensor(peek_saliency_action(pano_maps, state_object.idx, self.delta_M, self.delta_N, state_object.delta_to_act))
                    act = target_actions.contiguous().view(-1, 1)
                    log_prob_act = None
                    entropy = None

                elif self.policy.actorType == 'const_action':
                    act = target_actions.clone()
                    log_prob_act = None
                    entropy = None

                actions_taken[:, t] = act[:, 0]

                # ---- Rotate the view of the state and collect environment rewards for this transition ----
                reward_expert = state_object.rotate(act[:, 0])
                reward_expert = torch.Tensor(reward_expert)
                if self.iscuda:
                    reward_expert = reward_expert.cuda()

                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg_expert = (self.R_avg_expert * self.avg_count_expert + reward_expert.sum())/(self.avg_count_expert + batch_size)
                    self.avg_count_expert += batch_size

                reward_all.append(reward_expert*self.reward_scale_expert)
                log_prob_act_all.append(log_prob_act)
                entropy_all.append(entropy)
                lookahead_pred_all.append(lookahead_pred)

        return reward_all, baseline_all, log_prob_act_all, entropy_all, classifier_activations_all, lookahead_pred_all, lookahead_gt_all, visited_idxes, actions_taken

    def update_policy(self, reward_all, baseline_all, log_prob_act_all, entropy_all, classifier_activations_all, class_labs, lookahead_pred_all, lookahead_gt_all):
        """
        This function will take the rewards, baselines, log probabilities, entropies, classifier
        activations and lookahead predictions. It will update policy using REINFORCE and the
        classifiers using classification criterion.
        INPUTS:
            reward_all: list of T-1 Tensors containing reward for each batch at each time step
            baseline_all: list of T-1 Tensors containing baseline for each batch at each time step
            log_prob_act_all: list of T-1 logprobs Variables of each transition of batch
            entropy_all: list of T-1 entropy Variables for each transition of batch
            classifier_activations_all: list of T classifier Variables predictions
            class_labs: Variable of ground truth classifications
            lookahead_pred_all: list of T-1 lookahead prediction Variables
            lookahead_gt_all: list of T-1 lookahead ground truth Variables
        """
        # ---- Setup initial values ----
        batch_size = classifier_activations_all[0].size(0)
        R = torch.zeros(batch_size) # Reward accumulator
        B = torch.zeros(batch_size) # Baseline accumulator
        loss = Variable(torch.Tensor([0]))
        if self.iscuda:
            loss = loss.cuda()
            R = R.cuda()
            B = B.cuda()

        # ---- Classification loss computation ----
        for t in range(self.T):
            loss = loss + self.criterion_classification(classifier_activations_all[t], class_labs)

        clsf_loss = loss.data.cpu()
        pg_loss = 0
        ent_loss = 0
        la_loss = 0

        # ---- REINFORCE loss based on T-1 transitions ----
        # Note: This will automatically be ignored when self.T = 1
        for t in reversed(range(self.T-1)):
            if self.policy.actorType == 'actor':
                # A one sample MC estimate of Q[t]
                R = R + reward_all[t]
                # B - an estimate of V[t] when no critic is present
                B = B + baseline_all[t] + self.R_avg_expert * self.reward_scale_expert
                # Compute the advantage
                adv = R - B
                # PG loss
                loss_term_1 = - (log_prob_act_all[t]*Variable(adv, requires_grad=False)*self.reward_scale).sum()/batch_size
                pg_loss += loss_term_1.data.cpu()
                # Entropy loss, maximize entropy
                loss_term_2 = - self.lambda_entropy*entropy_all[t].sum()/batch_size
                ent_loss += loss_term_2.data.cpu()
                loss = loss + loss_term_1 + loss_term_2

        # ---- Lookahead loss ----
        for t in range(self.T-1):
            loss_term_3 = - self.lambda_la*F.cosine_similarity(lookahead_pred_all[t], lookahead_gt_all[t], dim=1).sum()/batch_size
            loss = loss + loss_term_3
            la_loss += loss_term_3.data.cpu()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.optimizer.step()

        return clsf_loss, pg_loss, ent_loss, la_loss
