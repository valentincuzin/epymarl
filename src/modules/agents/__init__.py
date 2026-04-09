from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent
from .gnn_agent import GNNAgent, GNNAgentV2
from .gnn_rnn_agent import GnnRnnAgent
from .rnn_gnn_agent import RnnGnnAgent
from .egcn_agent import EGCNAgent, EGCNAgentV2
from .egcn_v2_agent import EGCN_HAgent, EGCN_OAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent


REGISTRY = {}
REGISTRY["mlp"] = MLPAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["gnn"] = GNNAgent
REGISTRY["gnn-2"] = GNNAgentV2
REGISTRY["gnn_rnn"] = GnnRnnAgent
REGISTRY["rnn_gnn"] = RnnGnnAgent
REGISTRY["egcn"] = EGCNAgent
REGISTRY["egcn-2"] = EGCNAgentV2
REGISTRY["egcn-h"] = EGCN_HAgent
REGISTRY["egcn-o"] = EGCN_OAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
