"""Smoke tests for the polymarket-vision-bot scaffold."""
from __future__ import annotations

import os

import pandas as pd
import pytest


def test_config_loads():
    from config.settings import config

    assert config.vision.chain_id == 111222333
    assert config.vision.usdc_decimals == 18


def test_imports():
    from data import (  # noqa: F401
        DataCleaner,
        MarketTelemetry,
        PolymarketDataLoader,
        label_outcome,
    )
    from features import (  # noqa: F401
        PolymarketELO,
        PolymarketFeatureEngineer,
        TripleLayerFeatures,
        add_external_signal_features,
        claude_analyze_market_context,
        compute_four_factors_stream,
        compute_overlap_features,
        compute_stream_fatigue_features,
    )
    from models import (  # noqa: F401
        PolymarketHybridPredictor,
        build_ensemble,
        prepare_model_data,
        train_and_evaluate,
    )
    from vision import VisionHistorical, VisionTestnetClient, VisionTrader  # noqa: F401


def test_abi_present():
    from config.settings import config

    assert (config.ABI_DIR / "Vision.json").exists()


def test_triple_layer_math():
    from features import TripleLayerFeatures

    external = {"yes": 0.6, "no": 0.4}
    vision = {"yes": 0.5, "no": 0.5}
    result = TripleLayerFeatures.compute_divergence_features(external, vision)
    assert "kl_div_ext_vision" in result
    assert result["max_divergence"] > 0


def test_streak_computation():
    from features import PolymarketFeatureEngineer

    fn = getattr(PolymarketFeatureEngineer, "_compute_streak", None)
    if not isinstance(fn, staticmethod) and not callable(fn):
        pytest.skip("_compute_streak not exposed")

    series = pd.Series([1, 1, 0, 0, 1])
    out = fn(series) if callable(fn) else fn.__func__(series)
    assert len(out) == len(series)
    assert out.iloc[-1] == 1


def test_label_outcome():
    from data import label_outcome

    df = pd.DataFrame(
        {
            "CATEGORY": ["politics", "politics", "sports", "sports"],
            "OUTCOME": [1, 0, 1, 0],
        }
    )
    out = label_outcome(df, threshold_quantile=0.5)
    assert "HITS_GOAL" in out.columns
    assert set(out["HITS_GOAL"].unique()).issubset({0, 1, True, False})


@pytest.mark.skipif(
    not os.getenv("POLYMARKET_NETWORK_TESTS"), reason="network tests disabled"
)
def test_polymarket_network():
    from data import PolymarketDataLoader

    loader = PolymarketDataLoader(categories=["politics"])
    raw = loader.load_all(lookback_days=7)
    assert raw is not None
