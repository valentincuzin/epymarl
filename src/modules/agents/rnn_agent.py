# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for _ in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.hidden_dim))
            input_shape = args.hidden_dim
        self.base = nn.Sequential(*self.fc_layers)

        self.rnn_layers = []
        for _ in range(args.n_layers_rnn):
            self.rnn_layers.append(nn.GRUCell(input_shape, args.hidden_dim))
            input_shape = args.hidden_dim
        self.rnn = nn.ModuleList(self.rnn_layers)

        self.act_prob = nn.Sequential(
            nn.LayerNorm(args.hidden_dim) if args.layer_norm else [],
            nn.Linear(args.hidden_dim, args.n_actions),
        )
        print(
            f"\n\nDEBUG: total number of PARAMETERS for RNNAgent: {sum(p.numel() for p in self.parameters())} #####\n\n"
        )

    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        self.hidden_states = []
        for _ in range(self.args.n_layers_rnn):
            self.hidden_states.append(
                param.new_zeros(1, self.args.hidden_dim)
                .unsqueeze(0)
                .expand(batch_size, n_agents, -1)
            )  # bav
        return self.hidden_states

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        for n in range(self.args.n_layers_rnn):
            hidden_state = hidden_states[n]
            h_rnn = hidden_state.reshape(-1, self.args.hidden_dim)
            h = self.rnn[n](x, h_rnn)
            hidden_states[n] = h
            x = h
        q = self.act_prob(h)
        return q, hidden_states
