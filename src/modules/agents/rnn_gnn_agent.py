# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn

# PYG
from torch_geometric.nn import GATv2Conv, MessagePassing

from utils.gnn_utils import batch_from_dense_to_ptg


class RnnGnnAgentBase(nn.Module):
    def __init__(self, input_shape, args):
        super(RnnGnnAgentBase, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)

        self.rnn = nn.GRUCell(input_shape, args.mem_dim)

        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(
            args.mem_dim,
            args.gnn_dim,
            edge_dim=3 if self.args.edge_attr else None,
            residual=self.args.residual_gat,
        )

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.mem_dim)

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        h_in = hidden_states.reshape(-1, self.args.mem_dim)
        h = self.rnn(x, h_in)
        z = self._communication_process(inputs, h)
        return z, h

    def _communication_process(self, inputs, x):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = self.gnns(
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr if self.args.edge_attr else None,
        )
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs


class RnnGnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RnnGnnAgent, self).__init__()
        self.rnn_gnn_base = RnnGnnAgentBase(input_shape, args)
        self.act_prob = nn.Sequential(
            nn.ReLU(), nn.Linear(args.gnn_dim, args.n_actions)
        )
        print(
            f"\n--- RnnGnnAgentBase {sum(p.numel() for p in self.parameters())} parameters --- \n\n",
            self,
            "\n\n",
        )

    def forward(self, inputs, hidden_states):
        z, h = self.rnn_gnn_base.forward(inputs, hidden_states)
        q = self.act_prob(z)
        return q, h

    def get_parent(self):
        return self.rnn_gnn_base

    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        return self.rnn_gnn_base.init_hidden(batch_size, n_agents)
