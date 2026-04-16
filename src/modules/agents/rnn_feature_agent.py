import torch.nn as nn


class RNNFeatureAgent(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, input_shape, args):
        nn.Module.__init__(self)
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.h_dim)
        self.rnn = nn.GRUCell(args.h_dim, args.h_dim)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.h_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = nn.functional.relu(self.fc1(inputs))
        h = self.rnn(x, hidden_state.reshape(-1, self.args.h_dim))
        return None, h