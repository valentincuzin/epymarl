from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent
from .gnn_agent import GNNAgent, GNNAgentV2
from .gnn_rnn_agent import GnnRnnAgent
from .rnn_gnn_agent import RnnGnnAgent
from .egcn_agent import EGCNAgent
from .tgn_agent import TGNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["gnn_v2"] = GNNAgentV2  # TODO to remove
REGISTRY["gnn_rnn"] = GnnRnnAgent
REGISTRY["rnn_gnn"] = RnnGnnAgent
REGISTRY["egcn"] = EGCNAgent
REGISTRY["tgn"] = TGNAgent

REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
