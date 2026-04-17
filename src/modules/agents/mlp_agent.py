# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn

class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.h_dim))
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)
        self.act_prob = nn.Linear(input_shape, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for MLPAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    # not used in this agent architecture
    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.h_dim)

    def forward(self, inputs, hidden_state=None):
        h = self.base(inputs)
        q = self.act_prob(h)
        return q, None
