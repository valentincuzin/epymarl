REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

from .ltscg_controller import LTSCG_GraphMAC
from .dicg_controller import DICGraphMAC
from .roland_controller import ROLANDMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["Ltscg_mac"] = LTSCG_GraphMAC
REGISTRY["dicg_mac"] = DICGraphMAC
REGISTRY["roland_mac"] = ROLANDMAC