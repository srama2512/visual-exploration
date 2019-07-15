import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.autograd import Variable
from base.common import *

class BasePolicy(nn.Module):
    """
    Base policy class
    """
    def __init__(self, opts):
        # ---- Settings for policy network ----
        super(BasePolicy, self).__init__()
        # Panorama operation settings
        self.M = opts.M
        self.N = opts.N
        self.A = opts.A
        self.C = opts.num_channels
        # Whether elevation on azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Whether azimuth, elevation are known to the sensor or not
        self.knownElev = opts.knownElev
        self.knownAzim = opts.knownAzim
        # Normalization settings
        self.mean = opts.mean
        self.std = opts.std
        # Network settings
        self.iscuda = opts.iscuda
        self.rnn_hidden_size = 256
        self.baselineType = opts.baselineType # Can be average or critic
        if not(opts.baselineType == 'critic' or opts.baselineType == 'average'):
            raise ValueError('baselineType %s does not exist!'%(opts.baselineType))
        self.act_full_obs = opts.act_full_obs
        self.critic_full_obs = opts.critic_full_obs
        self.actorType = opts.actorType
        if hasattr(opts, 'const_act'):
            self.const_act = opts.const_act

        # ---- Create the Policy Network ----
        # The input size for location embedding / proprioception stack
        input_size_loc = 2 # Relative camera position
        if self.knownAzim:
            input_size_loc += 1
        if self.knownElev:
            input_size_loc += 1

        # (1) Sense - image: Takes in BxCx32x32 image input and converts it to Bx256 matrix
        self.sense_im = nn.Sequential( # BxCx32x32
                            nn.Conv2d(self.C, 32, kernel_size=5, stride=1, padding=2), # Bx32x32x32
                            nn.MaxPool2d(kernel_size=3, stride=2), # Bx32x15x15
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2), # Bx32x15x15
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=3, stride=2), # Bx32x7x7
                            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # Bx64x7x7
                            nn.ReLU(inplace=True),
                            nn.AvgPool2d(kernel_size=3, stride=2), # Bx64x3x3
                            View(-1, 576),
                            nn.Linear(576, 256),
                            nn.ReLU(inplace=True)
                        )

        # (2) Sense - proprioception stack: Converts proprioception inputs to 16-D vector
        self.sense_pro = nn.Sequential(
                            nn.Linear(input_size_loc, 16),
                            nn.ReLU(inplace=True)
                         )

        # (3) Fuse: Fusing the outputs of (1) and (2) to give 256-D vector per image
        self.fuse = nn.Sequential( # 256+16
                        nn.Linear(272, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 256), # Bx256
                        nn.BatchNorm1d(256)
                    )

        # (4) Aggregator: View aggregating LSTM
        self.aggregate = nn.LSTM(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1)

        # (5) Act module: Takes in aggregator hidden state + other inputs to
        # produce probability distribution over actions
        if self.actorType == 'actor':
            if not self.act_full_obs:
                input_size_act = self.rnn_hidden_size + 2 # Add the relative positions
                # Optionally feed in elevation, azimuth
                if opts.actOnElev:
                    input_size_act += 1
                if opts.actOnAzim:
                    input_size_act += 1
                if opts.actOnTime:
                    input_size_act += 1
                self.act = nn.Sequential( # self.rnn_hidden_size
                                nn.Linear(input_size_act, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, self.A)
                           )
            else:
                # Fully observed actor
                input_size_act = self.rnn_hidden_size + 5 # delta_elev, delta_azim, elev, azim, time
                input_size_act += 256 # Panorama encoded
                self.act_fuse = nn.Sequential( # BNM x 256
                                            nn.Linear(256,128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            View(-1, self.N*self.M*128),
                                            nn.Linear(self.N*self.M*128, 256),
                                            nn.BatchNorm1d(256),
                                      )

                self.act =  nn.Sequential( # self.rnn_hidden_size + 5 + 256
                                nn.Linear(input_size_act, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, self.A)
                           )
            # Assuming Critic not needed without an Actor
            # (5b) Critic module
            if opts.baselineType == 'critic':
                if not self.critic_full_obs:
                    # Partially observed critic
                    self.critic = nn.Sequential( # self.rnn_hidden_size
                                        nn.Linear(input_size_act, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 1)
                                   )
                else:
                    # Fully observed critic
                    input_size_critic = self.rnn_hidden_size + 5# delta_elev, delta_azim, elev, azim, time 
                    input_size_critic += 256 # Panorama encoded
                    self.critic_fuse = nn.Sequential( # BNM x 256
                                            nn.Linear(256, 128),
                                            nn.BatchNorm1d(128),
                                            nn.ReLU(inplace=True),
                                            View(-1, self.N*self.M*128),
                                            nn.Linear(self.N*self.M*128, 256),
                                            nn.BatchNorm1d(256),
                                      )
                    self.critic = nn.Sequential( # self.rnn_hidden_size+5+256
                                        nn.Linear(input_size_critic, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 1)
                                   )

        # ---- Initialize parameters according to specified strategy ----
        if opts.init == 'xavier':
            init_strategy = ixvr
        elif opts.init == 'normal':
            init_strategy = inrml
        else:
            init_strategy = iunf

        self.sense_im = initialize_sequential(self.sense_im, init_strategy)
        self.sense_pro = init_strategy(self.sense_pro)
        self.fuse = initialize_sequential(self.fuse, init_strategy)
        self.aggregate = init_strategy(self.aggregate)
        if self.actorType == 'actor':
            self.act = initialize_sequential(self.act, init_strategy)
            if self.act_full_obs:
                self.act_fuse = initialize_sequential(self.act_fuse, init_strategy)
            if self.baselineType == 'critic':
                self.critic = initialize_sequential(self.critic, init_strategy)
                if self.critic_full_obs:
                    self.critic_fuse = initialize_sequential(self.critic_fuse, init_strategy)

    def forward(self, x, hidden=None):
        """
        x consists of the integer proprioception *pro* and the view *im*. It can optionally
        absolute elevation *elev*, absolute azimuth *azim* and time *time* as well. If the
        critic is fully observed, the input must include the batch of full panoramas *pano*.
        """
        # ---- Setup the initial setup for forward propagation ----
        batch_size = x['im'].size(0)

        if hidden is None:
            hidden = [Variable(torch.zeros(1, batch_size, self.rnn_hidden_size)), # hidden state: (num_layers, batch_size, hidden size)
                      Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))] # cell state  :(num_layers, batch_size, hidden size)
            if self.iscuda:
                hidden[0] = hidden[0].cuda()
                hidden[1] = hidden[1].cuda()

        # ---- Sense the inputs ----
        xin = x
        x1 = self.sense_im(x['im'])
        x2 = self.sense_pro(x['pro'])

        if self.actorType == 'actor':
            # ---- Create the inputs for the actor ----
            if self.actOnElev:
                xe = x['elev']
            if self.actOnAzim:
                xa = x['azim']
            if self.actOnTime:
                xt = x['time']

        x = torch.cat([x1, x2], dim=1)
        # ---- Fuse the representations ----
        x = self.fuse(x)

        # ---- Update the belief state about the panorama ----
        # Note: input to aggregate lstm has to be seq_length x batch_size x input_dims
        # Since we are feeding in the inputs one by one, it is 1 x batch_size x 256
        x, hidden = self.aggregate(x.view(1, *x.size()), hidden)
        # Note: hidden[0] = h_n , hidden[1] = c_n
        if self.actorType == 'actor':
            if not self.act_full_obs:
                act_input = hidden[0].view(batch_size, -1)
                # Concatenate the relative change
                act_input = torch.cat([act_input, xin['pro'][:, :2]], dim=1)
                if self.actOnElev:
                    act_input = torch.cat([act_input, xe], dim=1)
                if self.actOnAzim:
                    act_input = torch.cat([act_input, xa], dim=1)
                if self.actOnTime:
                    act_input = torch.cat([act_input, xt], dim=1)
            else:
                # If fully observed actor
                act_input = hidden[0].view(batch_size, -1)
                # BxNxMx32x32 -> BNMx256
                pano_encoded = self.sense_im(xin['pano'].view(-1, self.C, 32, 32))
                # BNMx256 -> Bx256
                pano_fused = self.act_fuse(pano_encoded)
                act_input = torch.cat([act_input, xin['pro'][:, :2], xin['elev'], \
                                       xin['azim'], xin['time'], pano_fused], dim=1)

            # ---- Predict the action propabilities ----
            probs = F.softmax(self.act(act_input), dim=1)

            # ---- Predict the value for the current state ----
            if self.baselineType == 'critic' and not self.critic_full_obs:
                values = self.critic(act_input).view(-1)
            elif self.baselineType == 'critic' and self.critic_full_obs:
                critic_input = hidden[0].view(batch_size, -1)
                # BxNxMxCx32x32 -> BNMx256
                pano_encoded = self.sense_im(xin['pano'].view(-1, self.C, 32, 32))
                # BNMx256 ->  Bx256
                pano_fused = self.critic_fuse(pano_encoded)
                critic_input = torch.cat([critic_input, xin['pro'][:, :2], xin['elev'], \
                                          xin['azim'], xin['time'], pano_fused], dim=1)
                values = self.critic(critic_input).view(-1)
            else:
                values = None
        else:
            probs = None
            act_input = None
            values = None

        return probs, hidden, values
