# code adapted from https://github.com/proroklab/HetGPPO

import torch
import torch.nn as nn
import torch.nn.functional as F

# PYG
import torch_geometric
from torch_geometric.nn import GCNConv, GATv2Conv, Sequential
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from functools import partial


# TODO à priori à virer, car simplifier dans batch_from_dense_to_ptg
# def get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
#     if topology == "full":
#         adjacency = torch.ones(n_agents, n_agents, device=device, dtype=torch.long)
#         edge_index, _ = torch_geometric.utils.dense_to_sparse(adjacency)
#         if not self_loops:
#             edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
#     elif topology == "empty":
#         if self_loops:
#             edge_index = (
#                 torch.arange(n_agents, device=device, dtype=torch.long)
#                 .unsqueeze(0)
#                 .repeat(2, 1)
#             )
#         else:
#             edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
#     elif topology == "from_pos":
#         edge_index = None
#     else:
#         raise ValueError(f"Topology {topology} not supported")

#     return edge_index

def batch_from_dense_to_ptg(x, batch_size, args) -> torch_geometric.data.Batch:
    x = x.reshape(-1, x.shape[-1])
    
    # 1. Récupérer les positions + velocité des noeuds à partir de x
        # if pos is not None:
        #     pos = pos.view(-1, pos.shape[-1])
        # if vel is not None:
        #     vel = vel.view(-1, vel.shape[-1])

    # géré les batchs comme ils faut: 1 graph par batch, à priori c'est bon ça
    b = torch.arange(batch_size, device=x.device)

    graphs = torch_geometric.data.Batch()
    graphs.ptr = torch.arange(0, (batch_size + 1) * args.n_agents, args.n_agents)
    graphs.batch = torch.repeat_interleave(b, args.n_agents)
    graphs.x = x
    # graphs.pos = pos
    # graphs.vel = vel
    graphs.edge_attr = None

    if args.comm_range == -1 or pos is None:  # TODO ici récupérer la bonne shape, puis faire des graphs complet
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = torch.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * args.n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(graphs.pos, batch=graphs.batch, r=args.comm_range, loop=False)
    graphs = graphs.to(x.device)
    # TODO: prove the improvment of this component
    # Add relative coordonate and distance in edge_attr in all the graph
    if pos is not None:
        graphs = torch_geometric.transforms.Cartesian(norm=False)(graphs)
        graphs = torch_geometric.transforms.Distance(norm=False)(graphs)
    # TODO prove the improvment of this component
    # Create relative velocity
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs

class _RelVel(BaseTransform):
    """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

    def __init__(self):
        pass

    def forward(self, data):
        (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

        cart = vel[row] - vel[col]
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        if pseudo is not None:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
        return data
