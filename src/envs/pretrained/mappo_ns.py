import torch as th
import torch.nn as nn

from modules.agents.rnn_agent import RNNAgent
from types import SimpleNamespace as SN



class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = 3
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList(
            [RNNAgent(input_shape, args) for _ in range(self.n_agents)]
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)


class MAPPO_NS():

    def __init__(self, actor_input_dim, actor_output_dim, hidden):
        args = {
            "hidden_dim": hidden,
            "n_actions": actor_output_dim,
        }
        args = SN(**args)
        self.policy = RNNNSAgent(actor_input_dim, args)
        self.hidden_states = self.policy.init_hidden().unsqueeze(0).expand(1, -1, -1)  # bav

    def step(self, obs):
        obs = th.Tensor(obs)
        action, self.hidden_states = self.policy(obs, self.hidden_states)
        action = action.argmax(dim=1)
        return tuple(action.cpu().numpy())

    def load_params(self, params):
        new_state = {}

        # update keys
        for k, v in params.items():
            parts = k.split(".", 3)  # ["agents", "<idx>", "<sub>", "rest..."]
            idx = int(parts[1])
            if idx in (10, 11, 12):
                new_idx = idx - 10  # 10->0, 11->1, 12->2
                rest = k.split(".", 2)[2]
                new_key = f"agents.{new_idx}.{rest}"
                new_state[new_key] = v
        self.policy.load_state_dict(new_state)
