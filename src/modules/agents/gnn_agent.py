# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn

from torch_geometric.nn import GATv2Conv, MessagePassing, Sequential, LayerNorm

from utils.gnn_utils import batch_from_dense_to_ptg

class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
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

        self.act_prob = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(2*args.hidden_dim) if args.layer_norm else [],
            nn.Linear(2*args.hidden_dim, args.n_actions)
        )
        print(f"\n--- GNNAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)
    
    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, None
    
    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr)

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
            self.fc_layers.append(nn.Linear(input_shape, args.hidden_dim))
            self.fc_layers.append(nn.ReLU())
            if args.layer_norm:
                self.fc_layers.append(nn.LayerNorm(args.hidden_dim))
            input_shape = args.hidden_dim
        self.base = nn.Sequential(*self.fc_layers)
        # comm modules:
        self.gnns: MessagePassing = Sequential(
            "x, edge_index", [
                (GATv2Conv(
                args.hidden_dim, 
                2*args.hidden_dim, 
                edge_dim=3, 
                residual=True), "x, edge_index -> x"),
                nn.ReLU(),
                (GATv2Conv(
                2*args.hidden_dim,
                2*args.hidden_dim,
                edge_dim=3,
                residual=True), "x, edge_index -> x"),
                nn.ReLU()
            ]
        )

        self.act_prob = nn.Sequential(
            nn.LayerNorm(2*args.hidden_dim),
            nn.Linear(2*args.hidden_dim, args.n_actions)
        )
        print(f"\n--- GNNAgentV2 {sum(p.numel() for p in self.parameters())} parameters --- \n\n", self, "\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, 2*self.args.hidden_dim)
    
    def forward(self, inputs, hidden_state=None):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, None
    
    def _communication_process(self, raw_inputs, x):
        graphs = self._select_communication(raw_inputs)
        graphs.x = x
        h = self.gnns(graphs.x, graphs.edge_index, graphs.edge_attr)

        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs
