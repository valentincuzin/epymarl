# code adapted from https://github.com/wendelinboehmer/dcg

from functools import partial

import torch.nn as nn
import torch.functional as F
from torch_geometric.nn import GATv2Conv, MessagePassing, Sequential
from utils.gnn_utils import batch_from_dense_to_ptg, print_graph, create_gif, attach_att
from functorch import make_functional_with_buffers


class GNNAgentBase(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgentBase, self).__init__()
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

    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h, graphs = self._communication_process(inputs, x)
        return h, None, graphs

    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h, att = self.gnns(
            graphs.x,
            graphs.edge_index,
            graphs.edge_attr if self.args.edge_attr else None,
            return_attention_weights=True
        )
        graphs = attach_att(graphs, att)
        return h, graphs

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs


class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args

        self.gnn_base = GNNAgentBase(input_shape, args)

        self.act_prob = nn.Sequential(
            nn.ReLU(), nn.Linear(args.gnn_dim, args.n_actions)
        )
        print(
            f"\n--- GNNAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n",
            self,
            "\n\n",
        )

    # not used in this agent architecture
    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.h_dim)

    def forward(self, inputs, hidden_state=None, fast_weights=None):
        if fast_weights:  # deterior performance for me...
            func, meta_weights, meta_buffers = make_functional_with_buffers(
                self.gnn_base
            )
            assert len(meta_weights) == len(fast_weights), (
                f"fast weights pass has not the expected len: {len(fast_weights)} instead of {len(meta_weights)}"
            )
            base_forward = partial(func, fast_weights, meta_buffers)
        else:
            base_forward = self.gnn_base.forward

        h, _, graphs = base_forward(inputs)
        q = self.act_prob(h)
        return q, None, graphs 
    

    def get_parent(self):
        return self.gnn_base


class GNNAgentV2(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgentV2, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())

            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)
        # comm modules:
        self.gnns: MessagePassing = Sequential(
            "x, edge_index",
            [
                (
                    GATv2Conv(
                        input_shape,
                        args.gnn_dim,
                        edge_dim=3 if self.args.edge_attr else None,
                        residual=self.args.residual_gat,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
                (
                    GATv2Conv(
                        args.gnn_dim,
                        args.gnn_dim,
                        edge_dim=3 if self.args.edge_attr else None,
                        residual=self.args.residual_gat,
                    ),
                    "x, edge_index -> x",
                ),
                nn.ReLU(),
            ],
        )

        self.act_prob = nn.Linear(args.gnn_dim, args.n_actions)
        print(
            f"\n--- GNNAgentV2 {sum(p.numel() for p in self.parameters())} parameters --- \n\n",
            self,
            "\n\n",
        )

    # not used in this agent architecture
    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.h_dim)

    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, None

    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
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
