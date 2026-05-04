# code adapted from https://github.com/wendelinboehmer/dcg

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# PYG
from torch_geometric.nn import GATv2Conv, MessagePassing, SimpleConv

from utils.gnn_utils import batch_from_dense_to_ptg

class TGNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TGNAgent, self).__init__()
        self.args = args
        raw_input_shape = input_shape

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.h_dim))
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)

        self.first_aggr_msg = SimpleConv(aggr=args.aggr, combine_root="self_loop")
        
        self.rnn = nn.GRUCell(input_shape, args.mem_dim)

        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(args.mem_dim+(raw_input_shape if self.args.skipfeats else 0), 
                                              args.gnn_dim, 
                                              edge_dim=3 if self.args.edge_attr else None,
                                              residual=self.args.residual_gat,
                                              dropout=self.args.dropout_gat,
                                              heads=self.args.n_heads_gat,
                                              concat=False)

        self.act_prob = nn.Sequential(
            nn.LayerNorm( args.gnn_dim) if args.layer_norm else nn.Identity(),
            nn.Linear(args.gnn_dim, args.n_actions)
        )
        print(f"\n--- TGNAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        self.hidden_states = (
            param.new_zeros(1, self.args.mem_dim)
            .unsqueeze(0)
            .expand(batch_size, n_agents, -1)
        )  # bav
        return self.hidden_states

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        graphs = self._select_communication(inputs)
        graphs.x = x
        x = self.first_aggr_msg(graphs.x, graphs.edge_index)
        h_in = hidden_states.reshape(-1, self.args.mem_dim)
        h = self.rnn(x, h_in)
        if self.args.skipfeats:
            h_skip = th.cat((h, inputs), dim=1)
        else:
            h_skip = h
        z = self._communication_process(inputs, h_skip)
        q = self.act_prob(z)
        return q, h

    def _communication_process(self, inputs, x):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = F.relu(self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr if self.args.edge_attr else None))
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
