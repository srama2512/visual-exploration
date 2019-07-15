import torch.nn.functional as F
import torch.nn as nn
import torch

from torch.autograd import Variable
from simple_rnn import SimpleRNN
from base.common import *

# Not using BasePolicy since the architecture uses features instead of images
class Policy(nn.Module):
    # The overall policy class
    def __init__(self, opts):
        # ---- Settings for policy network ----
        super(Policy, self).__init__()
        # Panorama operation settings
        self.M = opts.M
        self.N = opts.N
        self.A = opts.A
        self.F = opts.F
        # Whether elevation on azimuth are fed to act module or not
        self.actOnElev = opts.actOnElev
        self.actOnAzim = opts.actOnAzim
        self.actOnTime = opts.actOnTime
        # Network settings
        self.iscuda = opts.iscuda
        self.rnn_hidden_size = 256
        self.baselineType = opts.baselineType # Can be average
        self.rnn_type = opts.rnn_type
        if not(opts.baselineType == 'average'):
            raise ValueError('baselineType %s does not exist!'%(opts.baselineType))
        self.actorType = opts.actorType
        if hasattr(opts, 'const_act'):
            self.const_act = opts.const_act
        self.normalize_hidden = opts.normalize_hidden
        self.nonlinearity = opts.nonlinearity # relu or tanh
        # ---- Create the Policy Network ----
        # The input size for location embedding / proprioception stack
        input_size_loc = 2 # Relative camera position
        # The input size for the actor module, if there is an actor module
        if self.actorType == 'actor':
            input_size_act = self.rnn_hidden_size
            # Optionally feed in elevation, azimuth, time
            if opts.actOnElev:
                input_size_act += 1
            if opts.actOnAzim:
                input_size_act += 1
            if opts.actOnTime:
                input_size_act += 1

        input_size_lookahead = self.rnn_hidden_size + 4

        # (1) Sense - image: Takes in BxF image vector input
        self.sense_im = nn.Sequential(nn.Dropout(p=opts.featDropout))

        # (2) Sense - motion stack: Converts motion inputs to 16-D vector
        self.sense_motion = nn.Sequential(
                            nn.Linear(input_size_loc, 16),
                            nn.ReLU(inplace=True)
                         )

        # (3) Fuse: Fusing the outputs of (1) and (2) to give 256-D vector per image
        list_of_modules = []
        list_of_modules.append(nn.Linear(self.F + 16, 256))
        if opts.addExtraLinearFuse:
            list_of_modules.append(nn.ReLU(inplace=True))
            list_of_modules.append(nn.Linear(256, 256)) # Bx256
        list_of_modules.append(nn.BatchNorm1d(256))
        list_of_modules.append(nn.Dropout(opts.combineDropout))

        self.fuse = nn.Sequential(*list_of_modules)

        # (4) Aggregator: View aggregating RNN / LSTM
        if opts.nonlinearity == 'relu':
            if self.rnn_type == 0:
                self.aggregate = SimpleRNN(input_size=256, hidden_size=self.rnn_hidden_size, nonlinearity='relu')
            elif self.rnn_type == 1:
                self.aggregate = nn.LSTM(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1, nonlinearity='relu')
            else:
                self.aggregate = nn.RNN(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1, nonlinearity='relu')
        elif opts.nonlinearity == 'tanh':
            if self.rnn_type == 0:
                self.aggregate = SimpleRNN(input_size=256, hidden_size=self.rnn_hidden_size)
            elif self.rnn_type == 1:
                self.aggregate = nn.LSTM(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1)
            else:
                self.aggregate = nn.RNN(input_size=256, hidden_size=self.rnn_hidden_size, num_layers=1)
        else:
            raise ValueError('%s nonlinearity does not exist!'%(opts.nonlinearity))

        # (5) Act module: Takes in aggregator hidden state + other inputs to produce probability 
        #                 distribution over actions
        if self.actorType == 'actor': 
            self.act = nn.Sequential( # self.rnn_hidden_size + 2 (or) 16 + 2
                            nn.Linear(input_size_act, self.A),
                            nn.Hardtanh(min_val=0, max_val=1, inplace=True)
                       )

        # (6) Lookahead module
        self.lookahead = nn.Sequential( # self.rnn_hidden_size + 2(delta) + 2(proprio)
                            nn.Linear(input_size_lookahead, 100),
                            nn.ReLU(inplace=True),
                            nn.Linear(100, 256),
                            nn.ReLU(inplace=True)
                         )

        # ---- Initialize parameters according to specified strategy ----
        if opts.init == 'xavier':
            init_strategy = ixvr
        elif opts.init == 'normal': 
            init_strategy = inrml
        else:
            init_strategy = iunf

        self.sense_im = initialize_sequential(self.sense_im, init_strategy)
        self.sense_motion = init_strategy(self.sense_motion)
        self.fuse = initialize_sequential(self.fuse, init_strategy)
        self.aggregate = init_strategy(self.aggregate)
        if self.actorType == 'actor':
            self.act = initialize_sequential(self.act, init_strategy)
        self.lookahead = initialize_sequential(self.lookahead, init_strategy)

    def forward(self, x, hidden=None):
        """
        x consists of the integer motion *delta*, view *im* and proprioception *pro*.
        It can optionally time *time* as well.
        """
        # ---- Setup the initial setup for forward propagation ----
        batch_size = x['im'].size(0)

        if hidden is None:
            if self.rnn_type == 0:
                hidden = Variable(torch.zeros(batch_size, self.rnn_hidden_size)) # hidden state: batch_size x hidden size
            elif self.rnn_type == 1:
                hidden = [Variable(torch.zeros(1, batch_size, self.rnn_hidden_size)), # hidden state: num_layers x batch_size x hidden size
                      Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))] # cell state  : num_layers x batch_size x hidden_size
            else:
                hidden = Variable(torch.zeros(1, batch_size, self.rnn_hidden_size))

            if self.iscuda:
                if self.rnn_type == 0 or self.rnn_type == 2:
                    hidden = hidden.cuda()
                else:
                    hidden[0] = hidden[0].cuda()
                    hidden[1] = hidden[1].cuda()

        # ---- Sense the inputs ----
        xin = x
        x1 = self.sense_im(x['im'])
        x2 = self.sense_motion(x['delta'])

        # ---- Fuse the representations ----
        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)

        # ---- Update the belief state about the panorama ----
        if self.rnn_type == 0:
            hidden = self.aggregate(x, hidden)
        elif self.rnn_type == 1:
            # Note: input to aggregate lstm has to be seq_length x batch_size x input_dims
            _, hidden = self.aggregate(x.view(1, *x.size()), hidden)
            # Note: hidden[0] = h_n , hidden[1] = c_n
        else:
            _, hidden = self.aggregate(x.view(1, *x.size()), hidden)

        if self.actorType == 'actor':
            if self.rnn_type == 0 or self.rnn_type == 2:
                act_input = hidden.view(batch_size, -1)
            else:
                act_input = hidden[0].view(batch_size, -1)

            if self.actOnElev:
                act_input = torch.cat([act_input, xin['pro'][:, 0].contiguous().view(-1, 1)], dim=1)
            if self.actOnAzim:
                act_input = torch.cat([act_input, xin['pro'][:, 1].contiguous().view(-1, 1)], dim=1)
            if self.actOnTime:
                act_input = torch.cat([act_input, xin['time']], dim=1)

            # ---- Predict the action propabilities ----
            # Note: adding 1e-8 because self.act(act_input) to handle zero activations
            probs = F.normalize(self.act(act_input) + 1e-8, p=1, dim=1)
        else:
            probs = None
            act_input = None

        # ---- Perform lookahead ----
        if self.rnn_type == 0 or self.rnn_type == 2:
            lookahead_input = hidden.view(batch_size, -1)
        else:
            lookahead_input = hidden[0].view(batch_size, -1)
        lookahead_input = torch.cat([lookahead_input, xin['delta'], xin['pro']], dim=1)
        lookahead_pred = self.lookahead(lookahead_input)

        return probs, hidden, lookahead_pred
