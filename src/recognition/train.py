"""
Script to train recognition policies
"""

import os
import sys
import pdb
import json
import copy
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
from arguments import get_args
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

def train(opts):
    """
    Training function - trains an agent for a fixed number of epochs
    """
    # Set number of actions
    opts.A = opts.delta_M * opts.delta_N
    # Set random seeds
    set_random_seeds(opts.seed)
    # Create actions mapping
    count_act = 0
    opts.act_to_delta = {}
    opts.delta_to_act = {}
    for i in range(-(opts.delta_N//2), opts.delta_N//2+1):
        for j in range(-(opts.delta_M//2), opts.delta_M//2+1):
            opts.act_to_delta[count_act] = (i, j)
            opts.delta_to_act[(i, j)] = count_act
            count_act += 1

    # Data loading
    if opts.expert_rewards:
        from data_loader import DataLoaderExpert as DataLoader
    else:
        from data_loader import DataLoaderSimple as DataLoader

    loader = DataLoader(opts)
    opts.frequent_class = loader.get_most_frequent_class()

    # Create the agent
    agent = Agent(opts)

    # Create tensorboard writer
    writer = SummaryWriter(log_dir=opts.save_path)
    if opts.load_model != '':
        # strict=False ensures that only the modules common to loaded_dict and agent.policy's state_dict are loaded.
        # Could potentially lead to errors being masked. Tread carefully!
        loaded_state = torch.load(opts.load_model)
        loaded_state_dict = loaded_state['policy_state_dict']
        # If action space changed, then load everything except the act module
        if opts.actorType == 'actor' and loaded_state_dict['act.0.weight'].shape != agent.policy.state_dict()['act.0.weight'].shape:
            policy_state_dict = agent.policy.state_dict()
            pretrained_dict = {key: val for key, val in loaded_state_dict.iteritems() if 'act.' not in key}
            policy_state_dict.update(pretrained_dict)
        else:
            policy_state_dict = loaded_state_dict

        agent.policy.load_state_dict(policy_state_dict, strict=False)

        for t in range(len(loaded_state['classifier_state_dict'])):
            agent.classifiers[t].load_state_dict(loaded_state['classifier_state_dict'][t])

    # Initiate statistics storage variables
    best_val_accuracy = 0
    train_err_history = []
    train_history = []
    val_history = []
    epoch_start = 0

    # To handle job eviction and restarts
    if os.path.isfile(os.path.join(opts.save_path, 'model_latest.net')):
        print('====> Resuming training from previous checkpoint')
        # undo most of the loading done before
        loaded_model = torch.load(os.path.join(opts.save_path, 'model_latest.net'))
        opts = loaded_model['opts']
        epoch_start = loaded_model['epoch'] + 1

        loader = DataLoader(opts)
        opts.frequent_class = loader.get_most_frequent_class()

        agent = Agent(opts)

        agent.policy.load_state_dict(loaded_model['policy_state_dict'])

        for i in range(len(agent.classifiers)):
            agent.classifiers[i].load_state_dict(loaded_model['classifier_state_dict'][i])

        best_val_accuracy = loaded_model['best_val_accuracy']
        train_err_history = loaded_model['train_err_history']
        train_history = loaded_model['train_history']
        val_history = loaded_model['val_history']

    # Set networks to train
    agent.policy.train()
    for t in range(opts.T):
        agent.classifiers[t].train()

    opts_train_copy = copy.deepcopy(opts)
    opts_train_copy.start_view = 0
    # Start training
    for epoch in range(epoch_start, opts.epochs):
        # Initialize epoch specific variables
        depleted = False
        train_err = 0
        train_count = 0
        iter_count = 0
        avg_la_loss = 0

        # Linear LR scheduling
        lr_curr = max(opts.lr - epoch*(opts.lr-opts.final_lr)/opts.saturate_epoch, opts.final_lr)
        agent.create_optimizer(lr_curr, opts.weight_decay, opts.momentum)
        while not depleted:
            # pano - BxNxMxF
            if opts.expert_rewards:
                pano, labs, pano_rewards, depleted = loader.next_batch('train')
                pano_maps = None
            else:
                pano, labs, depleted = loader.next_batch('train')
                pano_rewards = None
                pano_maps = None

            # Note: This batch size is the current batch size, not the global batch size. This varies
            # when you reach the boundary of the dataset.
            batch_size = pano.shape[0]
            start_idx = get_starts(opts.N, opts.M, batch_size, opts_train_copy.start_view)

            state = State(pano, labs, pano_rewards, start_idx, opts_train_copy)
            class_labs = torch.LongTensor(labs)
            if opts.iscuda:
                class_labs = class_labs.cuda()
            class_labs = Variable(class_labs)

            # Forward pass
            reward_all, baseline_all, log_prob_act_all, entropy_all, \
            classifier_activations_all, lookahead_pred_all, \
            lookahead_gt_all, visited_idxes, _ = agent.gather_trajectory(state, eval_opts=None, pano_maps=pano_maps)
            # Backward pass
            train_loss, pg_loss, ent_loss, la_loss = agent.update_policy(reward_all, baseline_all, log_prob_act_all, entropy_all, \
                                                                                     classifier_activations_all, class_labs, lookahead_pred_all, \
                                                                                     lookahead_gt_all)
            # Accumulate statistics
            avg_la_loss += la_loss * batch_size
            train_err += train_loss * batch_size
            train_count += batch_size
            iter_count += 1
        
        train_err /= train_count
        avg_la_loss /= train_count
        
        # Evaluate the agent after every epoch 
        train_accuracy, _, _ = evaluate(loader, agent, 'train', opts_train_copy)
        val_accuracy, predicted_activations, visited_idxes_val = evaluate(loader, agent, 'val', opts)
        # ================================================================
        #                      Visualization part
        # ================================================================
        # Note: visited_idxes_val = B-D list of T-D list of batch_size-D list of 2-D list
        vis_samples = min(3, len(predicted_activations))
        # Create empty canvas
        vis_tensors = torch.zeros((vis_samples * opts.batch_size, opts.T, opts.N, opts.M, 30, 30, 1))
        # fill in borders (only one vertical and horizontal border)
        vis_tensors[:, :, :, :, 0, :, :] = 1
        vis_tensors[:, :, :, :, :, 0, :] = 1
        for b1 in range(vis_samples):
            batch_size_curr = len(visited_idxes_val[b1][0])
            for b2 in range(batch_size_curr):
                for t in range(opts.T):
                    vis_tensors[b1*opts.batch_size + b2, t, visited_idxes_val[b1][t][b2][0], visited_idxes_val[b1][t][b2][1], :, :, :] = 0.5
        # Convert to B x T x N*10 x M*10 x 1
        vis_tensors = torch.Tensor.permute(vis_tensors, 0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, opts.T, opts.N*30, opts.M*30, 1)
        # Convert to B x T x 1 X 10N x 10M
        vis_tensors = torch.Tensor.permute(vis_tensors, 0, 1, 4, 2, 3).contiguous()
        vis_tensors_size = vis_tensors.size(0)
        # List of T x 1 x 10N x 10M tensors
        vis_tensors = [torchvision.utils.make_grid(vis_tensors[b], opts.T, padding=4, pad_value=1.0) \
                                                                            for b in range(vis_tensors_size)]
        for b in range(vis_tensors_size):
            writer.add_image('Validation Panorama #%d'%(b), vis_tensors[b], epoch+1)

        # ================================================================

        #scheduler.step(val_accuracy)
        # Write out statistics to tensorboard
        writer.add_scalar('data/train_error', train_err, epoch+1)
        writer.add_scalar('data/train_accuracy', train_accuracy, epoch+1)
        writer.add_scalar('data/val_accuracy', val_accuracy, epoch+1)
        writer.add_scalar('data/loookahead_loss', avg_la_loss, epoch+1)
       
        # Write out models and other statistics to torch format file
        train_err_history.append([epoch, train_err])
        train_history.append([epoch, train_accuracy])
        val_history.append([epoch, val_accuracy])
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            save_state = {
                            'epoch': epoch,
                            'policy_state_dict': agent.policy.state_dict(),
                            'classifier_state_dict':[classifier.state_dict() for classifier in agent.classifiers],
                            'optimizer': agent.optimizer.state_dict(),
                            'opts': opts, 
                            'best_val_accuracy': best_val_accuracy,
                            'train_err_history': train_err_history,
                            'train_history': train_history,
                            'val_history': val_history,
                         }

            torch.save(save_state, os.path.join(opts.save_path, 'model_best.net'))

        save_state = {
                        'epoch': epoch,
                        'classifier_state_dict':[classifier.state_dict() for classifier in agent.classifiers],
                        'policy_state_dict': agent.policy.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'opts': opts, 
                        'best_val_accuracy': best_val_accuracy,
                        'train_err_history': train_err_history,
                        'train_history': train_history,
                        'val_history': val_history,
                     }

        torch.save(save_state, os.path.join(opts.save_path, 'model_latest.net'))

        print('Epoch %d : Train loss: %9.6f    Train accuracy: %6.3f Val accuracy: %6.3f'%(epoch+1, train_err, train_accuracy, val_accuracy))

if __name__ == '__main__':

    opts = get_args()
    assert (opts.lr >= opts.final_lr), "Cannot have final_lr greater than lr!"

    train(opts)
