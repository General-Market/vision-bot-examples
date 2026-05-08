from __future__ import annotations

import anthropic
import pandas as pd

from config.settings import config
from features.claude_features import claude_synthesize_triple_layer
from features.triple_layer import TripleLayerFeatures
from vision.client import VisionTestnetClient


class TwitchHybridPredictor:
    def __init__(self, ml_model, scaler, feature_names: list[str]):
        self.ml_model = ml_model
        self.scaler = scaler
        self.feature_names = feature_names
        self.anthropic = anthropic.Anthropic()
        self.vision = VisionTestnetClient()
        self.triple = TripleLayerFeatures()

    def predict(
        self,
        stream_features: pd.DataFrame,
        channel: str,
        market_question: str,
        vision_market: dict | None = None,
        external_probs: dict | None = None,
    ) -> dict:
        X_scaled = self.scaler.transform(stream_features[self.feature_names])
        ml_proba = self.ml_model.predict_proba(X_scaled)[0]
        ml_result = {"no": float(ml_proba[0]), "yes": float(ml_proba[1])}

        vision_probs: dict | None = None
        if vision_market is not None:
            vision_probs = {
                "yes": vision_market["yes"],
                "no": vision_market["no"],
            }

        ext_probs = external_probs or {
            "yes": ml_result["yes"],
            "no": ml_result["no"],
        }

        if vision_probs is not None:
            divergence = TripleLayerFeatures.compute_divergence_features(
                ext_probs, vision_probs, ml_result
            )
        else:
            divergence = {}

        vision_liquidity = (
            vision_market.get("liquidity_usdc") if vision_market else None
        )

        claude_result = claude_synthesize_triple_layer(
            channel,
            market_question,
            ml_result,
            ext_probs,
            vision_probs,
            divergence,
            vision_liquidity,
        )

        return self._triple_combine(
            ml_result,
            ext_probs,
            vision_probs,
            claude_result,
            vision_liquidity,
        )

    @staticmethod
    def _triple_combine(
        ml_result: dict,
        ext_probs: dict,
        vision_probs: dict | None,
        claude_result: dict,
        vision_liquidity: float | None,
    ) -> dict:
        if "error" in claude_result:
            weights = {"ml": 1.0, "vision": 0.0, "ext": 0.0, "claude": 0.0}
        elif vision_probs is not None and (vision_liquidity or 0) > 10_000:
            weights = {"ml": 0.35, "vision": 0.35, "ext": 0.15, "claude": 0.15}
        elif vision_probs is not None and (vision_liquidity or 0) > 1_000:
            weights = {"ml": 0.40, "vision": 0.20, "ext": 0.20, "claude": 0.20}
        else:
            weights = {"ml": 0.50, "vision": 0.0, "ext": 0.25, "claude": 0.25}

        claude_probs = {
            "yes": float(claude_result.get("adjusted_yes", ml_result["yes"])),
            "no": float(claude_result.get("adjusted_no", ml_result["no"])),
        }

        combined: dict[str, float] = {}
        for k in ("yes", "no"):
            v = weights["ml"] * ml_result[k]
            v += weights["ext"] * float(ext_probs.get(k, ml_result[k]))
            if vision_probs is not None:
                v += weights["vision"] * float(vision_probs[k])
            v += weights["claude"] * claude_probs[k]
            combined[k] = v

        total = combined["yes"] + combined["no"]
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}

        predicted = max(combined, key=combined.get)

        return {
            "predicted_result": predicted,
            "probabilities": combined,
            "weights_used": weights,
            "confidence": claude_result.get("confidence", "unknown"),
            "expected_peak_viewers": claude_result.get(
                "expected_peak_viewers", 0
            ),
            "insight": claude_result.get("key_insight", ""),
            "risk": claude_result.get("risk_factor", ""),
            "source": "triple_hybrid",
        }


_ = config

__all__ = ["TwitchHybridPredictor"]
