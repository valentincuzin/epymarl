import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_shape, h_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
