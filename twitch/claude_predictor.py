"""Claude-as-tiebreaker predictor.

Starts from a base predictor's scores. For the top-K markets whose scores
sit closest to the threshold (the model is uncertain), call Claude with
a compact summary of the asset's recent behaviour and let it override.

Cheap: you only spend tokens on the marginal picks where the base model
is already ambivalent — everywhere else the base score stands.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import pandas as pd


class ClaudePredictor:
    name = "claude"

    def __init__(
        self,
        base_predictor,
        features_df: Optional[pd.DataFrame] = None,
        top_k: int = 20,
        threshold: float = 0.0,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.base = base_predictor
        self.features_df = features_df
        self.top_k = top_k
        self.threshold = threshold
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set — claude predictor needs it."
            )
        import anthropic

        self._client = anthropic.Anthropic()

    def _summary(self, asset_id: str, snap: dict) -> str:
        name = snap.get("name") or asset_id
        chg = snap.get("changePct") or 0
        val = snap.get("value") or 0
        feats = {}
        if self.features_df is not None and asset_id in self.features_df.index:
            r = self.features_df.loc[asset_id]
            feats = {
                "change_1h": float(r.get("change_1h", 0.0)),
                "change_6h": float(r.get("change_6h", 0.0)),
                "change_24h": float(r.get("change_24h", 0.0)),
                "vol_1h": float(r.get("vol_1h", 0.0)),
                "streak": int(r.get("streak", 0)),
                "slope_1h": float(r.get("slope_1h", 0.0)),
            }
        return (
            f"{asset_id} ({name}): current_value={val}, live_change_pct={chg}, "
            f"history={feats}"
        )

    def _call(self, summaries: list[tuple[str, str]]) -> dict[str, str]:
        self._ensure_client()
        bullets = "\n".join(f"- {line}" for _, line in summaries)
        prompt = (
            "You are a short-horizon quant for a Twitch-viewership prediction "
            "market on Vision L3. For each asset below, output a single token "
            "UP or DOWN predicting whether its value will be higher at the "
            "next 60-second tick. Reply with ONLY a JSON object mapping "
            "asset_id -> \"UP\" or \"DOWN\". No prose.\n\n" + bullets
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    return {}
        return {}

    def predict(self, markets, snapshot_by_id) -> list[float]:
        base_scores = self.base.predict(markets, snapshot_by_id)
        if not base_scores:
            return base_scores

        distances = [
            (i, abs(s - self.threshold)) for i, s in enumerate(base_scores)
        ]
        distances.sort(key=lambda t: t[1])
        marginal_idx = [i for i, _ in distances[: self.top_k]]

        summaries = [
            (
                markets[i].get("assetId", f"m_{i}"),
                self._summary(
                    markets[i].get("assetId", f"m_{i}"),
                    snapshot_by_id.get(markets[i].get("assetId"), {}),
                ),
            )
            for i in marginal_idx
        ]
        try:
            verdicts = self._call(summaries)
        except Exception as e:
            print(f"[claude] call failed: {e} — falling back to base scores")
            return base_scores

        scores = list(base_scores)
        for i in marginal_idx:
            aid = markets[i].get("assetId")
            v = verdicts.get(aid)
            if v == "UP":
                scores[i] = max(scores[i], self.threshold + 0.01)
            elif v == "DOWN":
                scores[i] = min(scores[i], self.threshold - 0.01)
        return scores
