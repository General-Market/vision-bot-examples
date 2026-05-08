"""Daily prediction pipeline wiring data -> features -> model -> Vision."""
from __future__ import annotations

import logging

from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import config
from data import DataCleaner, TwitchDataLoader, label_outcome
from features import (
    TwitchELO,
    TwitchFeatureEngineer,
    compute_four_factors_stream,
    compute_overlap_features,
    compute_stream_fatigue_features,
)
from models import prepare_model_data
from vision import VisionTestnetClient

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class TwitchPredictionPipeline:
    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels or config.twitch.channels
        self.loader = TwitchDataLoader(channels=self.channels)
        self.engineer = TwitchFeatureEngineer(window=config.model.rolling_window)
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss",
        )
        self.scaler = StandardScaler()
        self.vision = VisionTestnetClient()

    def run_daily(self) -> dict:
        logger.info("step 1/12 loading twitch data (lookback=%d)", config.twitch.lookback_days)
        try:
            raw = self.loader.load_all(lookback_days=config.twitch.lookback_days)
        except Exception as exc:
            logger.exception("twitch load failed: %s", exc)
            raise

        logger.info("step 2/12 cleaning %d raw rows", len(raw))
        clean = DataCleaner.clean(raw)

        logger.info("step 3/12 labeling outcomes (q=%.2f)", config.model.threshold_quantile)
        labeled = label_outcome(clean, threshold_quantile=config.model.threshold_quantile)

        logger.info("step 4/12 building stream features")
        featured = self.engineer.build_stream_features(labeled)

        logger.info("step 5/12 computing four factors")
        featured = compute_four_factors_stream(featured)

        logger.info("step 6/12 computing ELO")
        featured = TwitchELO(k=24, primetime_bonus=80).compute_elo_features(featured)

        logger.info("step 7/12 computing fatigue")
        featured = compute_stream_fatigue_features(featured)

        logger.info("step 8/12 computing overlap")
        featured = compute_overlap_features(featured)

        logger.info("step 9/12 preparing model data")
        X, y, feature_names = prepare_model_data(featured)

        logger.info("step 10/12 fitting scaler + xgb on %d rows, %d features", len(X), len(feature_names))
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        train_acc = self.model.score(X_scaled, y)
        logger.info("step 11/12 training accuracy: %.4f", train_acc)
        print(f"training_accuracy={train_acc:.4f}")

        logger.info("step 12/12 querying active twitch vision markets")
        try:
            active = self.vision.list_active_markets(source_prefix="twitch")
        except Exception as exc:
            logger.exception("vision rpc failed: %s", exc)
            raise
        print(f"active_twitch_markets={len(active)}")

        return {
            "streams_labeled": len(featured),
            "active_vision_markets": len(active),
            "feature_count": len(feature_names),
        }
