from .rnn_agent import RNNAgent
from .gnn_agent import GNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
