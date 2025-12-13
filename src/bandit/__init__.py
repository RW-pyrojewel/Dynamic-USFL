# src/bandits/__init__.py
"""
Bandit algorithms for dynamic split-point selection.

Currently includes:
- LinUCBDualCut: LinUCB-E style linear bandit over dual-cut actions d = (s1, s2).
"""

from .linucb_dualcut import (
    CutPair,
    BanditDecision,
    LinUCBDualCut,
    build_linucb_dualcut,
)

from .context import (
    profile_backbone_layers,
    build_static_context_for_cut
)

__all__ = [
    "CutPair",
    "BanditDecision",
    "LinUCBDualCut",
    "build_linucb_dualcut",
    "profile_backbone_layers",
    "build_static_context_for_cut",
]
