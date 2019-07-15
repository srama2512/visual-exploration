import torch
from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        # Assumes 1 layer RNN and that the input size is same as hidden size
        super(SimpleRNN, self).__init__()
        assert(input_size == hidden_size)
        self.hidden_size = hidden_size
        self.feedback = nn.Linear(hidden_size, hidden_size)
        if nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        else:
            raise ValueError('There is no nonlinearity called %s !'%(nonlinearity))

    def forward(self, x_t, h_t_1=None):
        if h_t_1 is None:
            batch_size = x_t.size(0)
            h_t_1 = torch.autograd.Variable(x_t.data.new(batch_size, self.hidden_size).zero_())

        output = x_t + self.feedback(h_t_1)
        output = self.nonlinearity(output)
        return output
