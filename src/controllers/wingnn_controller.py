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

class PartialAgent(nn.Module):
    def __init__(self, agent_base, prediction_head):
        super().__init__()
        self.agent_base = agent_base
        self.prediction_head = prediction_head

    def forward(self, prev_node_states, action_onehot, hidden_states, fast_weights):
        emb, _ = self.agent_base(prev_node_states, hidden_states)
        z = th.cat((emb, action_onehot), dim=1)
        z = self.prediction_head(z)
        return z, emb
    
    def next_forward(self, agent_inputs, hidden_states, fast_weights):
        z, _ = self.agent_base(agent_inputs, hidden_states)
        return z

# This multi-agent controller shares parameters between agents
class WINGNNMAC:
    def __init__(self, scheme, groups, args):
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

        self.partial_agent = PartialAgent(self.agent.get_parent(), self.prediction_head)
        self.model_init = None
        self.prev_node_states = None  # no previous state on day 0.
        # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

        self.lambd = 0.1
        self.sig_reg = SIGReg().cuda()

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def train_step(self, agent_inputs, last_actions, fast_weights):
        hidden_states = self.hidden_states.detach()
        last_actions = last_actions.detach()

        z, emb = self.partial_agent(self.prev_node_states, last_actions, hidden_states, fast_weights)
        y = self.partial_agent.next_forward(agent_inputs, hidden_states, fast_weights)
        mse_loss = F.mse_loss(z, y)
        sigreg_loss = self.sig_reg(emb)
        loss = mse_loss + self.lambd * sigreg_loss
        return loss

    def fine_tune(self, ep_batch, t, fast_weights) -> None:
        agent_inputs = self._build_inputs(ep_batch, t)
        last_actions = ep_batch["actions_onehot"][:, t - 1].reshape(-1, self.args.n_actions).cuda()

        loss = self.train_step(agent_inputs, last_actions, fast_weights)
        self.prev_node_states = agent_inputs.detach()
        grad = th.autograd.grad(loss, fast_weights)
        return grad, loss

    def forward(self, ep_batch, t, test_mode=False, fast_weights=None, S_dw=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        if self.prev_node_states is None:
            self.prev_node_states = agent_inputs.detach()
        avail_actions = ep_batch["avail_actions"][:, t]
        if not test_mode:
            t_grad, t_loss = self.fine_tune(ep_batch, t, fast_weights)
            S_dw = list(map(lambda p: self.args.beta * p[1] + (1 - self.args.beta) * p[0] * p[0], zip(t_grad, S_dw)))

            fast_weights = list(map(lambda p: p[1] - self.args.maml_lr / (th.sqrt(p[2]) + 1e-8) * p[0], zip(t_grad, fast_weights, S_dw)))

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, fast_weights)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        self.prev_node_states = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

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
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)  # shape: (batch*nagent, obs_dim)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
