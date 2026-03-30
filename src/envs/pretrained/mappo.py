import torch as th

from modules.agents.rnn_agent import RNNAgent
from types import SimpleNamespace as SN


class MAPPO():

    def __init__(self, actor_input_dim, actor_output_dim, hidden):
        args = {
            "hidden_dim": hidden,
            "n_actions": actor_output_dim,
        }
        args = SN(**args)
        self.policy = RNNAgent(actor_input_dim, args)
        self.hidden_states = self.policy.init_hidden()

    def step(self, obs):
        with th.no_grad():
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
            if idx == 12:
                new_idx = 0
                rest = k.split(".", 2)[2]
                new_key = rest
                new_state[new_key] = v
        self.policy.load_state_dict(new_state)
