import torch
import torch.nn as nn
import torch.nn.functional as F

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden, norm_in=True):
#         super(MLP, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         # create network layers
#         if norm_in:  # normalize inputs
#             self.in_fn = nn.BatchNorm1d(input_dim)
#             self.in_fn.weight.data.fill_(1)
#             self.in_fn.bias.data.fill_(0)
#         else:
#             self.in_fn = lambda x: x
#         self.fc1 = nn.Linear(input_dim, hidden)
#         self.fc2 = nn.Linear(hidden, hidden)
#         self.fc3 = nn.Linear(hidden, output_dim)

#     def forward(self, x):
#         print("x: ", x)
#         h = F.relu(self.fc1(self.in_fn(x)))
#         h = F.relu(self.fc2(h))
#         out = self.fc3(h)
#         return out

class MLPAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super(MLPAgent, self).__init__()
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)
        print(f"\n\nDEBUG: total number of PARAMETERS for MLPAgent: {sum(p.numel() for p in self.parameters())} #####\n\n")

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        q = self.fc3(h)
        return q, None

class DDPG():

    def __init__(self, actor_input_dim, actor_output_dim, hidden):
        self.policy = MLPAgent(actor_input_dim, actor_output_dim, hidden)

    def step(self, obs, explore=False):
        obs = torch.Tensor(obs)
        action = self.policy(obs)
        action = action.argmax(dim=1)
        return action.cpu().numpy()

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])