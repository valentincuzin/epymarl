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

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.hidden_dim))
            input_shape = args.hidden_dim
        self.base = nn.Sequential(*self.fc_layers)

        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(args.hidden_dim, 
                                              2*args.hidden_dim, 
                                              edge_dim=3,
                                              residual=True)
        self.rnn = nn.GRUCell(2*args.hidden_dim, 2*args.hidden_dim)

        self.act_prob = nn.Sequential(
            nn.LayerNorm(2*args.hidden_dim) if args.layer_norm else [],
            nn.Linear(2*args.hidden_dim, args.n_actions)
        )
        print(f"\n--- GnnRnnAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        h = self._communication_process(inputs, x, hidden_states)
        q = self.act_prob(h)
        return q, h

    def _communication_process(self, inputs, x, hidden_states):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = F.relu(self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr))
        # h = th.cat(  # skip connection like CD-GCN does
        #         (h, x), dim=1
        #     )
        h_in = hidden_states.reshape(-1, 2*self.args.hidden_dim)
        h = self.rnn(h, h_in)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
