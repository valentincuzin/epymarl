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

        # comm modules:
        self.gnns: MessagePassing  = GATv2Conv(input_shape, args.hidden_dim, edge_dim=3)
        self.rnn = nn.GRUCell(args.hidden_dim+input_shape, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for GnnRnnAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.gnns.lin_l.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_states):
        h = self._communication_process(inputs, hidden_states)
        q = self.fc2(h)
        return q, h

    def _communication_process(self, inputs, hidden_states):
        graphs = self._select_communication(inputs)
        h = F.relu(self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr))
        h = th.cat(  # skip connection like CD-GCN does
                (h, inputs), dim=1
            )
        h_in = hidden_states.reshape(-1, self.args.hidden_dim)
        h = self.rnn(h, h_in)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
