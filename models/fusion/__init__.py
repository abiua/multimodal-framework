# models/fusion/__init__.py
from .registry import FusionRegistry, BaseFusion, IdentityFusion
from .strategies import (
    GateInjectionFusion,
    CrossAttentionFusion,
    TokenMixerFusion,
)
from .temporal_consensus import BidirectionalCrossAttention
from .film_gate import FiLMGate
