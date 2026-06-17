REGISTRY = {}

from .basic_controller import BasicMAC
from .comm_controller import CommMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

from .dicg_controller import DICGraphMAC
from .roland_controller import ROLANDMAC
from .wingnn_controller import WINGNNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["comm_mac"] = CommMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["dicg_mac"] = DICGraphMAC
REGISTRY["roland_mac"] = ROLANDMAC
REGISTRY["wingnn_mac"] = WINGNNMAC