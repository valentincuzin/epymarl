from .rnn_agent import RNNAgent
from .gnn_agent import GNNAgent
from .gnn_agent2 import GNNAgent2
from .egcn_agent import EGCNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["gnn2"] = GNNAgent2
REGISTRY["egcn"] = EGCNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
