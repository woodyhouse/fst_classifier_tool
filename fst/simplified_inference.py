"""
Backward-compatible simplified inference entry.

This class now reuses HybridInferenceEngine so the simplified and hybrid
paths stay fully aligned.
"""
from __future__ import annotations

from fst.hybrid_inference import HybridInferenceEngine


class SimplifiedHybridInference(HybridInferenceEngine):
    """Compatibility alias for older test scripts."""

