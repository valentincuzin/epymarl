# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for _ in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)

        self.rnn = nn.GRUCell(input_shape, args.mem_dim)

        self.act_prob = nn.Linear(args.mem_dim, args.n_actions)
        print(
            f"\n\nDEBUG: total number of PARAMETERS for RNNAgent: {sum(p.numel() for p in self.parameters())} #####\n\n"
        )

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.mem_dim)

    def forward(self, inputs, hidden_state):
        x = self.base(inputs)
        h_rnn = hidden_state.reshape(-1, self.args.mem_dim)
        h_rnn = self.rnn(x, h_rnn)
        q = self.act_prob(h_rnn)
        return q, h_rnn
