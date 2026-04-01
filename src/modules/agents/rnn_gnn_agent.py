# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F

# PYG
from torch_geometric.nn import GATv2Conv, MessagePassing

from utils.gnn_utils import batch_from_dense_to_ptg

class RnnGnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RnnGnnAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, 2*args.hidden_dim)

        # comm modules:
        self.gnns: MessagePassing  = GATv2Conv(2*args.hidden_dim, args.hidden_dim, edge_dim=5)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for RnnGnnAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_states.reshape(-1, 2*self.args.hidden_dim)
        h_s = self.rnn(x, h_in)
        h = F.relu(self._communication_process(inputs, h_s))
        q = self.fc2(h)
        return q, h_s

    def _communication_process(self, inputs, h_s):
        graphs = self._select_communication(inputs)
        graphs.x = h_s
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
