# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F

# PYG
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATv2Conv, Sequential, MessagePassing
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree

from utils.gnn_utils import batch_from_dense_to_ptg

class GNNAgent2(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent2, self).__init__()
        self.args = args
        
        # comm modules:
        self.e = nn.Sequential(
            nn.Linear(input_shape, args.hidden_dim),
            nn.ReLU())
        self.gnns: MessagePassing  = GATv2Conv(args.hidden_dim, args.hidden_dim, edge_dim=5, residual=True)

        assert self.args.use_rnn, "please mark use_rnn for this DGNN model"
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for GNNAgent2: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.e[0].weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_states):
        h = self._communication_process(inputs, hidden_states)
        q = self.fc2(h)
        return q, h

    def _communication_process(self, inputs, hidden_states):
        x = self.e(inputs)
        h_in = hidden_states.reshape(-1, self.args.hidden_dim)

        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        
        graphs = self._select_communication(x)
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
