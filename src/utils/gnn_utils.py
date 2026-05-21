# code adapted from https://github.com/proroklab/HetGPPO

import torch as th

# PYG
import torch_geometric as pyg
from torch_geometric.nn.pool import radius_graph

import networkx as nx
import matplotlib.pyplot as plt

import imageio
import glob


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
    if args.comm_range == -1:  # no comm range
        adjacency = th.ones(
            args.n_agents, args.n_agents, device=x.device, dtype=th.long
        )
        edge_index, _ = pyg.utils.dense_to_sparse(adjacency)
        edge_index, _ = pyg.utils.remove_self_loops(
            edge_index
        )  # self-loops treat by add_self_loop option in methods to avoid double addition
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
        graphs.edge_index = radius_graph(
            graphs.pos, batch=graphs.batch, r=args.comm_range, loop=False
        )
    graphs = graphs.to(x.device)
    # old todo: prove the improvment of this component => better
    # Add relative coordonate and distance in edge_attr in all the graph
    if pos is not None:
        graphs = pyg.transforms.Cartesian(norm=False)(graphs)  # test to remove
        graphs = pyg.transforms.Distance(norm=False)(graphs)
    return graphs


def print_graph(graphs: pyg.data.Batch, batch_size: int, t: int, args):
    # retrive only the first graphs batch and create a Data object
    graph = pyg.data.Data()
    graph.x = graphs.x.view(batch_size, args.n_agents, graphs.x.shape[-1])[0, ...]
    graph.pos = graphs.pos.view(batch_size, args.n_agents, graphs.pos.shape[-1])[0, ...]
    graph.vel = graphs.vel.view(batch_size, args.n_agents, graphs.vel.shape[-1])[0, ...]
    # slice only node in range of batch_size
    graph.edge_index = pyg.utils.unbatch_edge_index(graphs.edge_index, graphs.batch)[0]
    G = pyg.utils.to_networkx(graph)
    colors = [n for n in G.nodes()]
    nx.draw(G, graph.pos.cpu().numpy(), node_size=20, arrowsize=5, node_color=colors)
    plt.savefig(f"results/graphs/{args.unique_token}-{t}.png")
    plt.clf()
    plt.close()


def create_gif(unique_token):

    images = sorted(glob.glob(f"results/graphs/{unique_token}-*.png"))
    frames = [imageio.imread(p) for p in images]
    imageio.mimsave(f"results/graphs/{unique_token}.gif", frames, duration=0.0004)


def _get_pos_from_x(x: th.Tensor, task_name: str):
    pos = None
    vel = None  # if there is no specific velocity, it's not a problem
    if "mpe2" in task_name:
        pos = x[:, :2]
        vel = x[:, 2:4]
    elif task_name.startswith("rware:"):
        pos = x[:, :2]
    return pos, vel
