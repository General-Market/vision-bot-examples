"""CLI entry point for the polymarket-vision bot."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import config
from data import DataCleaner, PolymarketDataLoader, label_outcome
from features import (
    PolymarketELO,
    PolymarketFeatureEngineer,
    add_external_signal_features,
    compute_four_factors_stream,
    compute_overlap_features,
    compute_stream_fatigue_features,
)
from models import PolymarketHybridPredictor, prepare_model_data
from pipeline import PolymarketPredictionPipeline
from vision import VisionTestnetClient


def cmd_pipeline(_args: argparse.Namespace) -> int:
    result = PolymarketPredictionPipeline().run_daily()
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_markets(_args: argparse.Namespace) -> int:
    client = VisionTestnetClient()
    markets = client.list_active_markets(source_prefix="poly_")
    for m in markets:
        batch_id = m.get("batch_id")
        name = m.get("name", "")
        yes = m.get("yes")
        no = m.get("no")
        liq = m.get("liquidity_usdc")
        print(f"{batch_id}, {name}, {yes}, {no}, {liq}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    market_id = args.market_id
    question = args.question
    category = args.category

    loader = PolymarketDataLoader(categories=[category])
    raw = loader.load_all(lookback_days=config.polymarket.lookback_days)
    clean = DataCleaner.clean(raw)
    labeled = label_outcome(clean, threshold_quantile=config.model.threshold_quantile)

    engineer = PolymarketFeatureEngineer(window=config.model.rolling_window)
    featured = engineer.compute_category_stats(labeled)
    featured = engineer.build_market_features(featured)
    featured = compute_four_factors_stream(featured)
    featured = PolymarketELO(k=24).compute_elo_features(featured)
    featured = compute_stream_fatigue_features(featured)
    featured = compute_overlap_features(featured)
    featured = add_external_signal_features(featured)

    X, y, feature_names = prepare_model_data(featured)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_scaled, y)

    predictor = PolymarketHybridPredictor(
        ml_model=model,
        scaler=scaler,
        feature_names=feature_names,
    )
    latest: pd.DataFrame = featured.tail(1)
    prediction = predictor.predict(
        market_features=latest,
        market_id=market_id,
        market_question=question,
    )
    print(json.dumps(prediction, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="polymarket-vision-bot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("pipeline", help="run the daily training pipeline")
    sub.add_parser("markets", help="list active polymarket vision markets")

    predict_p = sub.add_parser(
        "predict", help="predict a single market by id + question"
    )
    predict_p.add_argument("--market-id", required=True)
    predict_p.add_argument("--question", required=True)
    predict_p.add_argument(
        "--category", default="politics",
        help="Polymarket category to use as training universe",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "pipeline": cmd_pipeline,
        "markets": cmd_markets,
        "predict": cmd_predict,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
