from .rnn_agent import RNNAgent
from .gnn_rnn_agent import GnnRnnAgent
from .rnn_gnn_agent import RnnGnnAgent
from .egcn_agent import EGCNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["gnn_rnn"] = GnnRnnAgent
REGISTRY["rnn_gnn"] = RnnGnnAgent
REGISTRY["egcn"] = EGCNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
