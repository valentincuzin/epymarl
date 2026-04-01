# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, MessagePassing

from utils.gnn_utils import batch_from_dense_to_ptg

class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        # comm modules:
        self.gnns: MessagePassing  = GATv2Conv(args.hidden_dim, 2*args.hidden_dim, edge_dim=5)

        self.fc3 = nn.Linear(2*args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for GNNAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)
    
    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        h = F.relu(self._communication_process(inputs, x))
        q = self.fc3(h)
        return q, None
    
    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
