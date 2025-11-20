"""
DHC-Jamba Enhanced

Hybrid Transformer-Mamba architecture for spatial-temporal modeling.
"""

__version__ = "1.0.0"

from .core.model import DHCJambaModel, DHCJambaConfig
from .core.jamba import JambaModel, JambaConfig, JambaBlock
from .adapters.rl_policy_jamba import JambaRLPolicy, JambaRLValue, JambaRLActorCritic

__all__ = [
    'DHCJambaModel',
    'DHCJambaConfig',
    'JambaModel',
    'JambaConfig',
    'JambaBlock',
    'JambaRLPolicy',
    'JambaRLValue',
    'JambaRLActorCritic',
]
