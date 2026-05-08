"""SullyGnome viewer telemetry — rate-limited public endpoint."""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)


class ViewerTelemetry:
    BASE = "https://sullygnome.com/api"

    def fetch_channel_stream_history(self, channel: str) -> pd.DataFrame:
        cols = ["STREAM_START", "AVG_VIEWERS", "PEAK_VIEWERS", "FOLLOWERS_GAINED"]
        try:
            r = requests.get(
                f"{self.BASE}/channels/{channel}/streams/365", timeout=15
            )
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException as e:
            log.warning("SullyGnome fetch failed for %s: %s", channel, e)
            time.sleep(0.5)
            return pd.DataFrame(columns=cols)

        rows = []
        for item in payload.get("data", []) or []:
            rows.append({
                "STREAM_START": pd.to_datetime(
                    item.get("startdatetime"), errors="coerce"
                ),
                "AVG_VIEWERS": float(item.get("avgviewers") or 0),
                "PEAK_VIEWERS": float(item.get("maxviewers") or 0),
                "FOLLOWERS_GAINED": float(item.get("followers") or 0),
            })

        time.sleep(0.5)
        return pd.DataFrame(rows, columns=cols)
