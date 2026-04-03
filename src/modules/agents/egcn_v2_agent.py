# code adapted from https://github.com/wendelinboehmer/dcg

from typing import Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# PYG
from utils.gnn_utils import batch_from_dense_to_ptg
from torch_geometric.utils import to_dense_adj
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.typing import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import TopKPooling


class EGCN_HAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EGCN_HAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.egcn = EvolveGCNH(args.n_agents*args.batch_size, args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim+args.hidden_dim, args.n_actions)
        print(
            f"\n\nDEBUG: total number of PARAMETERS for EGCN_HAgent: {sum(p.numel() for p in self.parameters())} #####\n\n"
        )

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        h = self._communication_process(inputs, x)
        q = self.fc2(h)
        return q, h

    def _communication_process(self, inputs, x):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = self.egcn(graphs.x, graphs.edge_index)
        h = th.cat(
            (h, x), dim=1
        )
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs


class EGCN_OAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EGCN_OAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.egcn = EvolveGCNO(args.hidden_dim)

        self.fc2 = nn.Linear(args.hidden_dim+args.hidden_dim, args.n_actions)
        print(
            f"\n\nDEBUG: total number of PARAMETERS for EGCN_OAgent: {sum(p.numel() for p in self.parameters())} #####\n\n"
        )

    def init_hidden(self):
        # make hidden states on same device as model
        param = next(self.parameters())
        return param.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        h = self._communication_process(inputs, x)
        q = self.fc2(h)
        return q, h

    def _communication_process(self, inputs, x):
        graphs = self._select_communication(inputs)
        graphs.x = x
        h = self.egcn(graphs.x, graphs.edge_index)
        h = th.cat(
            (h, x), dim=1
        )
        return h

    def _select_communication(self, x):
        graphs = batch_from_dense_to_ptg(x, self.args.batch_size, self.args)
        return graphs

# Code from EvolveGCN-h https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcnh.py

class EvolveGCNH(nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_of_nodes: int,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = nn.Parameter(th.Tensor(1, in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)

    def reinitialize_weight(self):
        self.weight = None

    def _create_layers(self):

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = nn.GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: th.FloatTensor,
        edge_index: th.LongTensor,
        edge_weight: th.FloatTensor = None,
    ) -> th.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        print("X_tilde.shape :", X_tilde[0].shape)
        X_tilde = X_tilde[0][None, :, :]
        if self.weight is None:
            _, self.weight = self.recurrent_layer(X_tilde, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(X_tilde, self.weight)
        X = self.conv_layer(self.weight.squeeze(dim=0), X, edge_index, edge_weight)
        return X

# --- code from https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py ---

class EvolveGCNO(nn.Module):
    r"""An implementation of the Evolving Graph Convolutional without Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_
    Args:
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNO, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.initial_weight = nn.Parameter(th.Tensor(1, in_channels, in_channels))
        self.weight = None
        self._create_layers()
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.initial_weight)

    def reinitialize_weight(self):
        self.weight = None

    def _create_layers(self):

        self.recurrent_layer = nn.GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: th.FloatTensor,
        edge_index: th.LongTensor,
        edge_weight: th.FloatTensor = None,
    ) -> th.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.
        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """

        if self.weight is None:
            _, self.weight = self.recurrent_layer(self.initial_weight, self.initial_weight)
        else:
            _, self.weight = self.recurrent_layer(self.weight, self.weight)
        X = self.conv_layer(self.weight.squeeze(dim=0), X, edge_index, edge_weight)
        return X

class GCNConv_Fixed_W(MessagePassing):
    r"""The graph convolutional operator adapted from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, with weights not trainable.
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.
    Its node-wise formulation is given by:
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j
    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)
    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[th.Tensor, th.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, W: th.FloatTensor, x: th.Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> th.Tensor:
        """"""

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = th.matmul(x, W)

        # propagate_type: (x: th.Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out

    def message(self, x_j: th.Tensor, edge_weight: OptTensor) -> th.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
