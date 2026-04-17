# code adapted from https://github.com/wendelinboehmer/dcg

import torch as th
import torch.nn as nn

from torch.nn.parameter import Parameter
import math
from types import SimpleNamespace as SN

# PYG
from utils.gnn_utils import batch_from_dense_to_ptg
from torch_geometric.utils import to_dense_adj


class EGCNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EGCNAgent, self).__init__()
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
        egcn_args = {
            "feats_per_node": input_shape,
            "layer_1_feats": args.gnn_dim,
        }
        for n in range(args.n_g_layers):
            egcn_args[f"layer_{n}_feats"] = args.gnn_dim
        self.egcns = EGCN(egcn_args, activation=nn.ReLU(), skipfeats=args.skipfeats)

        self.act_prob = nn.Sequential(
            nn.LayerNorm(args.gnn_dim + (input_shape if args.skipfeats else 0))
            if args.layer_norm
            else nn.Identity(),
            nn.Linear(
                args.gnn_dim + (input_shape if args.skipfeats else 0), args.n_actions
            ),
        )
        print(
            f"\n--- EGCNAgent {sum(p.numel() for p in self.parameters())} parameters --- \n\n",
            self,
            "\n\n",
        )

    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        param = next(self.parameters())
        self.hidden_states = (
            param.new_zeros(1, self.args.gnn_dim)
            .unsqueeze(0)
            .expand(batch_size, n_agents, -1)
        )  # bav
        return self.hidden_states

    def forward(self, inputs, hidden_states):
        x = self.base(inputs)
        h = self._communication_process(inputs, x)
        q = self.act_prob(h)
        return q, h

    def _communication_process(self, inputs, x):
        graphs = self._select_communication(inputs)
        graphs.x = x
        dense_adj = to_dense_adj(
            graphs.edge_index,
            max_num_nodes=graphs.max_num_nodes,
        ).squeeze()
        h = self.egcns(dense_adj, graphs.x)
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs


# Code from EvolveGCN-o https://github.com/IBM/EvolveGCN/blob/master/egcn_o.py
class EGCN(nn.Module):
    def __init__(self, args: dict, activation, device="cpu", skipfeats=False):
        super().__init__()

        feats = list(args.values())
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            GRCU_args = SN(
                **{
                    "in_feats": feats[i - 1],
                    "out_feats": feats[i],
                    "activation": activation,
                }
            )

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def forward(self, A_list, Nodes_list, nodes_mask_list=None):
        node_feats = Nodes_list

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)  # , nodes_mask_list)
        out = Nodes_list
        if self.skipfeats:
            out = th.cat(
                (out, node_feats), dim=1
            )  # use node_feats.to_dense() if 2hot encoded input
        return out


class GRCU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        cell_args = SN()
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(
            th.empty(self.args.in_feats, self.args.out_feats)
        )
        self.reset_param(self.GCN_init_weights)

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, node_embs_list):  # ,mask_list):
        GCN_weights = self.GCN_init_weights
        node_embs = node_embs_list
        # first evolve the weights from the initial and use the new weights with the node_embs
        GCN_weights = self.evolve_weights(GCN_weights)  # ,node_embs,mask_list[t])
        node_embs = self.activation(A_list.matmul(node_embs.matmul(GCN_weights)))

        return node_embs


class mat_GRU_cell(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows, args.cols, nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows, args.cols, nn.Tanh())

        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q):  # ,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(th.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(th.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(th.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1.0 / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)

        return out


class TopK(nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(th.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1.0 / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices, self.k)

        tanh = nn.Tanh()

        if isinstance(node_embs, th.sparse.FloatTensor) or isinstance(
            node_embs, th.cuda.sparse.FloatTensor
        ):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()


def pad_with_last_val(vect, k):
    device = "cuda" if vect.is_cuda else "cpu"
    pad = th.ones(k - vect.size(0), dtype=th.long, device=device) * vect[-1]
    vect = th.cat([vect, pad])
    return vect
