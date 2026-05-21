# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F

# PYG
from torch_geometric.nn import GATv2Conv, MessagePassing

from utils.gnn_utils import batch_from_dense_to_ptg


class GnnRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GnnRnnAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())

            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)

        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(
            input_shape,
            args.gnn_dim,
            edge_dim=3 if self.args.edge_attr else None,
            residual=self.args.residual_gat,
        )
        self.rnn = nn.GRUCell(args.gnn_dim, args.mem_dim)

        self.act_prob = nn.Linear(args.mem_dim, args.n_actions)
        print(
            f"\n--- GnnRnnAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n",
            self,
            "\n\n",
        )

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.mem_dim)

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        h = self._communication_process(inputs, x, hidden_states)
        q = self.act_prob(h)
        return q, h

    def _communication_process(self, inputs, x, hidden_states):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = F.relu(
            self.gnns(
                graphs.x,
                graphs.edge_index,
                graphs.edge_attr if self.args.edge_attr else None,
            )
        )
        h_in = hidden_states.reshape(-1, self.args.mem_dim)
        h = self.rnn(h, h_in)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
