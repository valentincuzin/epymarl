from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy

class SIGReg(th.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = th.linspace(0, 3, knots, dtype=th.float32).cuda()
        dt = 3 / (knots - 1)
        weights = th.full((knots,), 2 * dt, dtype=th.float32).cuda()
        weights[[0, -1]] = dt
        window = th.exp(-t.square() / 2.0).cuda()
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = th.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    

class MlpProdDecoder(nn.Module):
    """Hadamard-product-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, h, u, v, adj=None):
        h_u = h[u]
        h_v = h[v]
        return self.net(h_u * h_v)

    def predict(self, h, u, v, adj=None):
        forward_res = self.forward(h, u, v)
        res = th.cat([th.sigmoid(forward_res)], dim=-1)
        return res


class PartialAgent(nn.Module):
    def __init__(self, agent_base, prediction_head):
        super().__init__()
        self.agent_base = agent_base
        self.prediction_head = prediction_head

    def forward(self, prev_node_states, action_onehot, hidden_states):
        emb, _ = self.agent_base(prev_node_states, hidden_states)
        z = th.cat((emb, action_onehot), dim=1)
        z = self.prediction_head(z)
        return z, emb
    
    def next_forward(self, agent_inputs, hidden_states):
        z, _ = self.agent_base(agent_inputs, hidden_states)
        return z

# This multi-agent controller shares parameters between agents
class ROLANDMAC(nn.Module):
    def __init__(self, scheme, groups, args):
        super(ROLANDMAC, self).__init__()
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.prediction_head = nn.Sequential(
            nn.Linear(args.gnn_dim+args.n_actions, args.gnn_dim),
            nn.ReLU(),
            nn.Linear(args.gnn_dim, args.gnn_dim),
        ).cuda()  # TODO test pretext task to predict future link as ROLAND DOES: 
           # suppose to have hard-attention / sparse adj matrix MlpProdDecoder(args.gnn_dim, args.gnn_dim)
        
        self.partial_agent = PartialAgent(self.agent.get_parent(), self.prediction_head)
        self.model_init = None
        self.prev_node_states = None  # no previous state on day 0.
        # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

        self.auc_hist = list()
        self.mrr_hist = list()

        # after not updating the best model for `tol` epochs, stop.
        self.meta_method = "moving_average"  # TODO: args.meta_method
        self.meta_alpha = 0.2

        self.lambd = 0.1
        self.sig_reg = SIGReg().cuda()
        self.horizon = 3

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        if not test_mode:
            agent_outputs = agent_outputs[0]
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def train_step(self, agent_inputs, last_actions):
        hidden_states = self.hidden_states.detach()
        last_actions = last_actions.detach()

        z, emb = self.partial_agent(self.prev_node_states, last_actions, hidden_states)
        y = self.partial_agent.next_forward(agent_inputs, hidden_states)
        mse_loss = F.mse_loss(z, y)
        sigreg_loss = self.sig_reg(emb)
        loss = mse_loss + self.lambd * sigreg_loss
        return loss

    # @th.no_grad()
    # def evaluate_step(self, agent_inputs):
    #     h = self.agent.rpz_forward(self.prev_node_states, self.hidden_states)
    #     z = self.prediction_head(h)
    #     loss = nn.MSELoss()(z, agent_inputs)
    #     return loss

    def fine_tune(self, ep_batch, t) -> None:
        agent_inputs = self._build_inputs(ep_batch, t)
        last_actions = ep_batch["actions_onehot"][:, t - 1].reshape(-1, self.args.n_actions).cuda()
        if self.prev_node_states is None:
            self.prev_node_states = agent_inputs.detach()
            return 0

        loss = self.train_step(agent_inputs, last_actions)

        self.prev_node_states = agent_inputs.detach()
        return loss


    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        t_loss = None
        if not test_mode:
            t_loss = self.fine_tune(ep_batch, t)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        if not test_mode:
            return agent_outs, t_loss
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden(batch_size, self.n_agents)
        self.prev_node_states = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac, _copy=False):
        if _copy:
            if other_mac.model_init is not None:
                self.agent.rnn_gnn_base.load_state_dict(copy.deepcopy(other_mac.model_init))
            else:
                self.agent.load_state_dict(other_mac.agent.state_dict())
        else:    
            self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat(
            [x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1
        )  # shape: (batch*nagent, obs_dim)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

@th.no_grad()
def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], th.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], th.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out