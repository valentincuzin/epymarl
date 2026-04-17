# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn

from torch_geometric.nn import GATv2Conv, MessagePassing, Sequential
from utils.gnn_utils import batch_from_dense_to_ptg, print_graph, create_gif

class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.h_dim))
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)
        # comm modules:
        self.gnns: MessagePassing = GATv2Conv(input_shape, 
                                              args.gnn_dim, 
                                              edge_dim=3 if self.args.edge_attr else None,
                                              residual=self.args.residual_gat,
                                              dropout=self.args.dropout_gat,
                                              heads=self.args.n_heads_gat,
                                              concat=False)

        self.act_prob = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(args.gnn_dim) if args.layer_norm else nn.Identity(),
            nn.Linear(args.gnn_dim, args.n_actions)
        )
        print(f"\n--- GNNAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    # not used in this agent architecture
    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        self.reset = True
        return param.new_zeros(1, self.args.gnn_dim)
    
    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, None
    
    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr if self.args.edge_attr else None)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs

class GNNAgentV2(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgentV2, self).__init__()
        self.args = args

        self.fc_layers = []
        for n in range(args.n_layers):
            self.fc_layers.append(nn.Linear(input_shape, args.h_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.h_dim))
            input_shape = args.h_dim
        self.base = nn.Sequential(*self.fc_layers)
        # comm modules:
        self.gnns: MessagePassing = Sequential(
            "x, edge_index", [
                (GATv2Conv(input_shape, 
                args.gnn_dim, 
                edge_dim=3 if self.args.edge_attr else None,
                residual=self.args.residual_gat,
                dropout=self.args.dropout_gat,
                heads=self.args.n_heads_gat,
                concat=False), "x, edge_index -> x"),
                nn.ReLU(),
                nn.LayerNorm(args.gnn_dim) if args.layer_norm else nn.Identity(),
                (GATv2Conv(args.gnn_dim, 
                args.gnn_dim, 
                edge_dim=3 if self.args.edge_attr else None,
                residual=self.args.residual_gat,
                dropout=self.args.dropout_gat,
                heads=self.args.n_heads_gat,
                concat=False), "x, edge_index -> x"),
                nn.ReLU(),
                nn.LayerNorm(args.gnn_dim) if args.layer_norm else nn.Identity(),
            ]
        )

        self.act_prob = nn.Linear(args.gnn_dim, args.n_actions)
        print(f"\n--- GNNAgentV2 {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    # not used in this agent architecture
    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        self.reset = True
        return param.new_zeros(1, self.args.gnn_dim)
    
    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, None
    
    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr if self.args.edge_attr else None)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
