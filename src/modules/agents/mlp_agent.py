# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for MLPAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, None

