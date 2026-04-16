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

        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(input_shape, 
                                              args.gnn_dim, 
                                              edge_dim=3 if self.args.edge_attr else None,
                                              residual=self.args.residual_gat,
                                              dropout=self.args.dropout_gat,
                                              heads=self.args.n_heads_gat,
                                              concat=False)
        self.rnn = nn.GRUCell(args.gnn_dim, args.mem_dim)

        self.act_prob = nn.Sequential(
            nn.LayerNorm(args.mem_dim) if args.layer_norm else [],
            nn.Linear(args.mem_dim, args.n_actions)
        )
        print(f"\n--- GnnRnnAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

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
        h = self._communication_process(inputs, hidden_states)
        q = self.act_prob(h)
        return q, h

    def _communication_process(self, inputs, hidden_states):
        graphs = self._select_communication(inputs)
        h = F.relu(self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr if self.args.edge_attr else None))
        # h = th.cat(  # skip connection like CD-GCN does
        #         (h, x), dim=1
        #     )
        h_in = hidden_states.reshape(-1, self.args.mem_dim)
        h = self.rnn(h, h_in)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
