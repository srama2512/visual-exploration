"""
Command-line arguments for training, evaluation of recognition policies
"""

import argparse

def str2bool(v):
    if v.lower() in ['y', 'yes', 't', 'true']:
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser()

    # Optimization options
    parser.add_argument('--h5_path', type=str, default='../data/SUN360/data.h5')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=7e-3)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--saturate_epoch', type=int, default=800)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--init', type=str, default='uniform', help='[ xavier | normal | uniform]')
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--combineDropout', type=float, default=0)
    parser.add_argument('--lambda_entropy', type=float, default=0, help='Coefficient of entropy term in loss')
    parser.add_argument('--lambda_la', type=float, default=1.5)
    parser.add_argument('--featDropout', type=float, default=0)
    parser.add_argument('--rnn_type', type=int, default=1, help='[ 0 - SimpleRNN | 1 - LSTM | 2 - Standard RNN]')
    parser.add_argument('--normalize_hidden', default=True, help='Are hidden vectors normalized before classification?')
    parser.add_argument('--nonlinearity', default='relu', help='[ relu | tanh ]')
    parser.add_argument('--optimizer_type', default='sgd', help='[ adam | sgd ]')

    # Agent options
    parser.add_argument('--dataset', type=int, default=0, help='[ 0: SUN360 | 1: ModelNet | 2: GERMS ]')
    parser.add_argument('--iscuda', type=str2bool, default=True)
    parser.add_argument('--actOnElev', type=str2bool, default=True)
    parser.add_argument('--actOnAzim', type=str2bool, default=False)
    parser.add_argument('--actOnTime', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--greedy', type=str2bool, default=False, help='enable greedy action selection during validation?')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | saved_trajectories | peek_saliency | const_action ]')
    parser.add_argument('--baselineType', type=str, default='average', help='[ average ]')
    parser.add_argument('--addExtraLinearFuse', type=str2bool, default=False)
    parser.add_argument('--trajectories_type', type=str, default='utility_maps', help='[ utility_maps | expert_trajectories ]')
    parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')

    # Environment options
    parser.add_argument('--T', type=int, default=5, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=12, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=12, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='azim neighborhood for actions')
    parser.add_argument('--delta_N', type=int, default=5, help='elev neighborhood for actions')
    parser.add_argument('--F', type=int, default=1024, help='Image feature size')
    parser.add_argument('--rewards_greedy', type=str2bool, default=False, help='enable greedy rewards?')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=True, help='wrap around the elevations when rotating?')
    parser.add_argument('--start_view', type=int, default=0, help='[0 - random , 1 - center , 2 - alternate]')
    parser.add_argument('--reward_scale', type=float, default=1, help='Scaling overall reward during REINFORCE')
    parser.add_argument('--expert_rewards', type=str2bool, default=False, help='Use rewards from expert agent?')
    parser.add_argument('--rewards_h5_path', type=str, default='', help='Reward file from expert agent')
    parser.add_argument('--reward_scale_expert', type=float, default=1e-0, help='Scaling for expert rewards if used')

    # Other options
    parser.add_argument('--num_classes', type=int, default=26)

    # Evaluation options
    parser.add_argument('--model_path', type=str, default='model_best.net')
    parser.add_argument('--eval_val', type=str2bool, default=False, help='evaluate on validation split?')
    parser.add_argument('--compute_all_times', type=str2bool, default=False, help='evaluate model at all time steps?')
    parser.add_argument('--average_over_time', type=str2bool, default=False, help='Average classifier activations at each time step?')

    args = parser.parse_args()

    return args
