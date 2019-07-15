from torch.autograd import Variable
from itertools import chain
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

from base.base_taskhead import BaseTaskHead
from base.base_agent import BaseAgent
from base.common import *

class TaskHead(BaseTaskHead):
    # The task prediction head
    def __init__(self, hidden_size):
        super(TaskHead, self).__init__(hidden_size, 4)

class Agent(BaseAgent):
    def __init__(self, opts, mode='train'):
        super(Agent, self).__init__(opts, mode=mode)

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
        pred_errs = []
        entropies = []
        hidden = None
        visited_idxes = []
        batch_size = state_object.batch_size
        start_idx = state_object.start_idx
        predictions_all = []
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

            # Note: hidden corresponds to the previous transition, where a new state was visited
            # and the belief was updated. probs and value correspond to the new transition, where the value
            # and action probabilities of the current state are estimated for PG update.
            probs, hidden, value = self.policy.forward(policy_input, hidden)
            predictions = self.task_head(hidden[0].view(batch_size, -1))
            # ---- Compute prediction loss (corresponding to the previous transition)----
            pred_err = state_object.loss_fn(predictions)
            predictions_all.append(predictions)

            # Prediction reward is obtained only at the final step
            # If there is only one step (T=1), then do not provide rewards
            # Note: This reward corresponds to the previous action
            if t < self.T-1 or t == 0:
                reward = torch.zeros(batch_size)
                if self.iscuda:
                    reward = reward.cuda()
            else:
                reward = -pred_err.data # Disconnects reward from future updates
                if self.baselineType == 'average' and self.mode == 'train':
                    self.R_avg = (self.R_avg * self.avg_count + reward.sum())/(self.avg_count + batch_size)
                    self.avg_count += batch_size
            if t > 0:
                rewards[t-1] += reward

            # There are self.T prediction errors as opposed to self.T-1 rewards
            pred_errs.append(pred_err)

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
                    act = torch.LongTensor(batch_size, 1).fill_(self.policy.const_act)
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

        return log_probs, pred_errs, rewards, entropies, values, visited_idxes, predictions_all, actions_taken

    def cuda(self):
        self.policy.cuda()
        self.task_head.cuda()

    def train(self):
        self.policy.train()
        self.task_head.train()

    def eval(self):
        self.policy.eval()
        self.task_head.eval()

    def load_state_dict(self, state, strict=True):
        self.policy.load_state_dict(state['policy_state_dict'], strict=strict)
        self.task_head.load_state_dict(state['task_head_state_dict'], strict=strict)

    def state_dict(self):
        return {'policy_state_dict': self.policy.state_dict(),
                'task_head_state_dict': self.task_head.state_dict()}

    def _create_policy(self, opts):
        self.policy = Policy(opts)

    def _create_task_head(self, opts):
        self.task_head = TaskHead(self.policy.rnn_hidden_size)

    def _get_parameters(self):
        return chain(self.policy.parameters(), self.task_head.parameters())
