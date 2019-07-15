"""
Command-line arguments for training, evaluation of pose-estimation policies
"""

import argparse

def str2bool(v):
    if v.lower() in ['y', 'yes', 't', 'true']:
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='data/sun360/sun360_processed.h5')
    parser.add_argument('--h5_path_unseen', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--init', type=str, default='xavier', help='[ xavier | normal | uniform ]')
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help='Entropy regularization')
    parser.add_argument('--critic_coeff', type=float, default=1e-2, help="coefficient for critic's loss term")
    parser.add_argument('--fix_decode', type=str2bool, default=False) 

    # Agent options
    parser.add_argument('--dataset', type=int, default=0, help='[ 0: SUN360 | 1: ModelNet ]')
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--actOnElev', type=str2bool, default=True)
    parser.add_argument('--actOnAzim', type=str2bool, default=False)
    parser.add_argument('--actOnTime', type=str2bool, default=True)
    parser.add_argument('--knownElev', type=str2bool, default=True)
    parser.add_argument('--knownAzim', type=str2bool, default=False)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=False, help='enable greedy action selection during validation?')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--memorize_views', type=str2bool, default=False)
    parser.add_argument('--mean_subtract', type=str2bool, default=True)
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | saved_trajectories | peek_saliency | const_action ]')
    parser.add_argument('--const_act', type=int, default=0, help='constant action to execute under const_action')
    parser.add_argument('--baselineType', type=str, default='average', help='[ average | critic ]')
    parser.add_argument('--act_full_obs', type=str2bool, default=False, help='Full observability for actor?')
    parser.add_argument('--critic_full_obs', type=str2bool, default=False, help='Full observability for critic?')
    parser.add_argument('--trajectories_type', type=str, default='utility_maps', help='[ utility_maps | expert_trajectories | saliency_scores ]')
    parser.add_argument('--utility_h5_path', type=str, default='', help='can be sidekick scores / saved trajectories / saliency scores')
    # Environment options
    parser.add_argument('--T', type=int, default=4, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=8, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=4, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='Number of movable azimuths')
    parser.add_argument('--delta_N', type=int, default=3, help='Number of movable elevations')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=False, help='wrap around the elevations when rotating?')
    parser.add_argument('--reward_scale', type=float, default=1, help='scaling for rewards')
    parser.add_argument('--expert_rewards', type=str2bool, default=False, help='Use rewards from expert agent?')
    parser.add_argument('--rewards_h5_path', type=str, default='', help='Reward file from expert agent')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-4, help='scaling for expert rewards if used')
    parser.add_argument('--start_view', type=int, default=0, help='[0 - random, 1 - center, 2 - alternate positions]')
    # Evaluation options
    parser.add_argument('--eval_val', type=str2bool, default=False, help='Evaluate on validation set?')
    parser.add_argument('--model_path', type=str, default='model_best.net', help='Path to model to be evaluated')
    
    args = parser.parse_args()

    return args
