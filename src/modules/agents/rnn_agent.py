# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.hidden_dim))
            input_shape = args.hidden_dim
        self.base = nn.Sequential(*self.fc_layers)
        self.rnn = nn.GRUCell(args.hidden_dim, 2*args.hidden_dim)
        self.act_prob = nn.Sequential(
            nn.LayerNorm(2*args.hidden_dim) if args.layer_norm else [],
            nn.Linear(2*args.hidden_dim, args.n_actions)
        )
        print(f"\n\nDEBUG: total number of PARAMETERS for RNNAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = self.base(inputs)
        h_in = hidden_state.reshape(-1, 2*self.args.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.act_prob(h)
        return q, h
