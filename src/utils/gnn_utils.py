# code adapted from https://github.com/proroklab/HetGPPO

import torch as th

# PYG
import torch_geometric as pyg
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj

# TODO à priori à virer, car simplifier dans batch_from_dense_to_ptg
# def get_edge_index(topology: str, self_loops: bool, n_agents: int, device: str):
#     if topology == "full":
        # adjacency = th.ones(n_agents, n_agents, device=device, dtype=th.long)
        # edge_index, _ = pyg.utils.dense_to_sparse(adjacency)
#         if not self_loops:
#             edge_index, _ = pyg.utils.remove_self_loops(edge_index)
#     elif topology == "empty":
#         if self_loops:
#             edge_index = (
#                 th.arange(n_agents, device=device, dtype=th.long)
#                 .unsqueeze(0)
#                 .repeat(2, 1)
#             )
#         else:
#             edge_index = th.empty((2, 0), device=device, dtype=th.long)
#     elif topology == "from_pos":
#         edge_index = None
#     else:
#         raise ValueError(f"Topology {topology} not supported")

#     return edge_index

def batch_from_dense_to_ptg(x, batch_size, args) -> pyg.data.Batch:
    if isinstance(x, list):
        x = th.tensor(x)
    x = x.reshape(-1, x.shape[-1])

    # 1. Récupérer les positions + velocité des noeuds à partir de x
    pos, vel = _get_pos_from_x(x, args.env_args["key"])

    # géré les batchs comme ils faut: 1 graph par batch, à priori c'est bon ça
    b = th.arange(batch_size, device=x.device)

    graphs = pyg.data.Batch()
    graphs.batch_size = batch_size
    graphs.ptr = th.arange(0, (batch_size + 1) * args.n_agents, args.n_agents)
    graphs.batch = th.repeat_interleave(b, args.n_agents)
    graphs.x = x
    graphs.max_num_nodes = x.shape[0]
    graphs.pos = pos
    graphs.vel = vel
    graphs.edge_attr = None

    if args.comm_range is None:
        graphs.edge_index = th.empty((2, 0), device=x.device, dtype=th.long)
        graphs = graphs.to(x.device)
        return graphs
    if args.comm_range == -1:
        adjacency = th.ones(args.n_agents, args.n_agents, device=x.device, dtype=th.long)
        edge_index, _ = pyg.utils.dense_to_sparse(adjacency)
        edge_index, _ = pyg.utils.remove_self_loops(edge_index)  # self-loops treat by add_self_loop option in methods to avoid double addition
        n_edges = edge_index.shape[1]
        # Tensor of shape [batch_size * n_edges]
        # in which edges corresponding to the same graph have the same index.
        batch = th.repeat_interleave(b, n_edges)
        # Edge index for the batched graphs of shape [2, n_edges * batch_size]
        # we sum to each batch an offset of batch_num * n_agents to make sure that
        # the adjacency matrices remain independent
        batch_edge_index = edge_index.repeat(1, batch_size) + batch * args.n_agents
        graphs.edge_index = batch_edge_index
    else:
        if pos is None:
            raise RuntimeError("from_pos topology needs positions as input")
        graphs.edge_index = pyg.nn.pool.radius_graph(graphs.pos, batch=graphs.batch, r=args.comm_range, loop=False)
    graphs = graphs.to(x.device)
    # TODO: prove the improvment of this component
    # Add relative coordonate and distance in edge_attr in all the graph
    if pos is not None:
        graphs = pyg.transforms.Cartesian(norm=False)(graphs)
        graphs = pyg.transforms.Distance(norm=False)(graphs)

    # TODO prove the improvment of this component
    # Create relative velocity
    if vel is not None:
        graphs = _RelVel()(graphs)

    return graphs

def _get_pos_from_x(x: th.Tensor, task_name):
    pos = None
    vel = None  # if there is no specific velocity, it's not a problem
    if task_name == "pz-mpe-simple-tag-v3":
        pos = x[:, 2:4]
        vel = x[:, :2]
    elif task_name.startswith("rware:"):
        pos = x[:, :2]
    return pos, vel

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
            data.edge_attr = th.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart
        return data
