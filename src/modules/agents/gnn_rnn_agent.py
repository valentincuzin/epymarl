# code adapted from https://github.com/wendelinboehmer/dcg
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# PYG
from torch_geometric.nn import GATv2Conv, MessagePassing

from utils.gnn_utils import batch_from_dense_to_ptg

class GnnRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GnnRnnAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)

        # comm modules:
        self.gnns: MessagePassing  = GATv2Conv(args.hidden_dim, args.hidden_dim, edge_dim=5)
        self.rnn = nn.GRUCell(args.hidden_dim+args.hidden_dim, 2*args.hidden_dim)

        self.fc2 = nn.Linear(2*args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for GnnRnnAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        h = self._communication_process(inputs, x, hidden_states)
        q = self.fc2(h)
        return q, h

    def _communication_process(self, inputs, x, hidden_states):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = F.relu(self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr))
        h = th.cat(  # skip connection like CD-GCN does
                (h, x), dim=1
            )
        h_in = hidden_states.reshape(-1, 2*self.args.hidden_dim)
        h = self.rnn(h, h_in)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
