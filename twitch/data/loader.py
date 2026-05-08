"""Twitch Helix loader: resolves channels and paginates past broadcasts."""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config.settings import config

log = logging.getLogger(__name__)

HELIX = "https://api.twitch.tv/helix"
_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


class TwitchDataLoader:
    def __init__(
        self,
        client_id: str | None = None,
        app_token: str | None = None,
        channels: list[str] | None = None,
    ):
        self.client_id = client_id or config.twitch.client_id
        self.app_token = app_token or config.twitch.app_token
        self.channels = channels or config.twitch.channels
        self.headers = {
            "Client-ID": self.client_id,
            "Authorization": f"Bearer {self.app_token}",
        }

    @staticmethod
    def _parse_duration(d: str) -> int:
        m = _DURATION_RE.fullmatch(d.strip())
        if not m:
            return 0
        h, mi, s = (int(x) if x else 0 for x in m.groups())
        return h * 3600 + mi * 60 + s

    def _get(self, url: str, params: dict) -> dict | None:
        try:
            r = requests.get(url, headers=self.headers, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning("Twitch GET failed %s: %s", url, e)
            return None

    def load_channel_streams(
        self, channel: str, lookback_days: int = 90
    ) -> pd.DataFrame:
        cols = [
            "CHANNEL", "USER_ID", "VIDEO_ID", "STREAM_START", "STREAM_END",
            "DURATION_SEC", "TITLE", "VIEW_COUNT_ARCHIVE", "LANGUAGE",
            "STREAM_DATE",
        ]
        user = self._get(f"{HELIX}/users", {"login": channel})
        time.sleep(0.1)
        if not user or not user.get("data"):
            return pd.DataFrame(columns=cols)
        user_id = user["data"][0]["id"]

        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        rows: list[dict] = []
        cursor: str | None = None

        while True:
            params = {
                "user_id": user_id,
                "type": "archive",
                "first": 100,
                "sort": "time",
            }
            if cursor:
                params["after"] = cursor
            page = self._get(f"{HELIX}/videos", params)
            time.sleep(0.1)
            if not page:
                break

            page_done = False
            for v in page.get("data", []):
                created = v.get("created_at", "")
                try:
                    start = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    start_naive = start.replace(tzinfo=None)
                except ValueError:
                    continue
                if start_naive < cutoff:
                    page_done = True
                    break

                dur = self._parse_duration(v.get("duration", ""))
                end = start_naive + timedelta(seconds=dur)
                rows.append({
                    "CHANNEL": channel,
                    "USER_ID": user_id,
                    "VIDEO_ID": v.get("id"),
                    "STREAM_START": start_naive,
                    "STREAM_END": end,
                    "DURATION_SEC": dur,
                    "TITLE": v.get("title", ""),
                    "VIEW_COUNT_ARCHIVE": v.get("view_count", 0),
                    "LANGUAGE": v.get("language", ""),
                    "STREAM_DATE": start_naive.date(),
                })

            cursor = page.get("pagination", {}).get("cursor")
            if page_done or not cursor:
                break

        return pd.DataFrame(rows, columns=cols)

    def load_all(self, lookback_days: int = 120) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for ch in self.channels:
            df = self.load_channel_streams(ch, lookback_days=lookback_days)
            print(f"[twitch] {ch}: {len(df)} streams")
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
