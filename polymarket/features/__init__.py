from .engineering import PolymarketFeatureEngineer
from .four_factors import compute_four_factors_stream
from .elo import PolymarketELO, TwitchELO  # TwitchELO alias retained
from .fatigue import compute_stream_fatigue_features
from .overlap import compute_overlap_features
from .external_signals import add_external_signal_features
from .claude_features import claude_analyze_market_context
from .triple_layer import TripleLayerFeatures

__all__ = [
    "PolymarketFeatureEngineer",
    "compute_four_factors_stream",
    "PolymarketELO",
    "TwitchELO",
    "compute_stream_fatigue_features",
    "compute_overlap_features",
    "add_external_signal_features",
    "claude_analyze_market_context",
    "TripleLayerFeatures",
]
