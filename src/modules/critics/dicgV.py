# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.attention_module import AttentionModule
from components.gcn_module import GCNModule

class DicgVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(DicgVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.residual = args.residual
        self.n_gcn_layers = args.n_g_layers
        self.dicg_layers = []
        self.dicg_emb_hid = args.dicg_emb_hid
        self.dicg_emb_dim = input_shape
        self.dicg_encoder = self._mlp(input_shape, self.dicg_emb_hid, self.dicg_emb_dim)
        self.dicg_layers.append(self.dicg_encoder)
        self.attention_layer = AttentionModule((self.dicg_emb_dim), attention_type='general')
        self.dicg_layers.append(self.attention_layer)
        self.gcn_layers = nn.ModuleList([GCNModule(in_features=(self.dicg_emb_dim), out_features=(self.dicg_emb_dim), bias=True, id=i) for i in range(self.n_gcn_layers)])
        self.dicg_layers.extend(self.gcn_layers)
        self.dicg_aggregator = self._mlp(input_shape, self.dicg_emb_hid, 1)

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t)
        embeddings_collection = []
        embeddings_0 = self.dicg_encoder.forward(inputs)
        embeddings_collection.append(embeddings_0)
        attention_weights = self.attention_layer.forward(embeddings_0)
        graph = th.ones((self.args.n_agents, self.args.n_agents), device=self.args.device)- th.eye(self.args.n_agents, device=self.args.device)
        graph = graph.repeat(bs, max_t ,1,1)

        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], graph*attention_weights)
            # print("embeddings_gcn",embeddings_gcn.shape)
            embeddings_collection.append(embeddings_gcn)

        if self.residual:
            dicg_emb = embeddings_collection[0] + embeddings_collection[-1]
        else:
            dicg_emb = embeddings_collection[-1]
        q = self.dicg_aggregator.forward(dicg_emb)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d

        layers.append(nn.Linear(dim, output))
        return (nn.Sequential)(*layers)