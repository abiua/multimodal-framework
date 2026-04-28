# models/fusion/__init__.py
from .registry import FusionRegistry, BaseFusion
from .strategies import (
    IdentityFusion,
    GateInjectionFusion,
    CrossAttentionFusion,
    TokenMixerFusion,
)
