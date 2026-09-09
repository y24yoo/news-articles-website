from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

FRED_BASE_URL = "https://api.stlouisfed.org/fred/"
SERIES_ENDPOINT = "category/series"
CHILDREN_ENDPOINT = "category/children"
OBSERVATION_ENDPOINT = "series/observations"


class FredApiError(RuntimeError):
    """Raised when the FRED API returns a non-200 response or an unexpected payload."""

    def __init__(self, status_code: int, payload: Any):
        super().__init__(f"FRED API error {status_code}: {payload}")
        self.status_code = status_code
        self.payload = payload


class FredService:
    """Thin async wrapper around the FRED HTTP API. No database or agent logic here."""

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient | None = None,
        lookback_years: int = 5,
    ):
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()
        self._lookback_years = lookback_years

    async def __aenter__(self) -> FredService:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _base_params(self) -> dict:
        return {"api_key": self._api_key, "file_type": "json"}

    def _observation_start(self) -> str:
        today = datetime.today()
        start = today.replace(year=today.year - self._lookback_years, month=1, day=1)
        return start.strftime("%Y-%m-%d")

    async def get_category_series(self, category_id: int) -> list:
        """Return the raw FRED series list for a category."""
        response = await self._client.get(
            FRED_BASE_URL + SERIES_ENDPOINT,
            params={**self._base_params(), "category_id": category_id},
        )
        response.raise_for_status()
        return response.json().get("seriess", [])

    async def get_category_children(self, category_id: int) -> list:
        """Return the raw FRED child categories for a category."""
        response = await self._client.get(
            FRED_BASE_URL + CHILDREN_ENDPOINT,
            params={**self._base_params(), "category_id": category_id},
        )
        response.raise_for_status()
        return response.json().get("categories", [])

    async def get_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> list:
        """Return [{"date": ..., "value": ...}, ...] for a series."""
        params = {
            **self._base_params(),
            "series_id": series_id,
            "observation_start": observation_start or self._observation_start(),
        }
        if observation_end is not None:
            params["observation_end"] = observation_end
        response = await self._client.get(FRED_BASE_URL + OBSERVATION_ENDPOINT, params=params)
        payload = response.json()
        if response.status_code != 200 or "observations" not in payload:
            raise FredApiError(response.status_code, payload)
        return [{"date": obs["date"], "value": obs["value"]} for obs in payload["observations"]]

    @staticmethod
    def get_series_metadata(raw_series: dict) -> dict:
        """Extract the searchable metadata fields from a raw FRED series object."""
        metadata = {
            "id": raw_series["id"],
            "title": raw_series["title"],
            "frequency": raw_series["frequency"],
            "units": raw_series["units"],
            "seasonal_adjustment": raw_series["seasonal_adjustment"],
        }
        notes = raw_series.get("notes")
        if notes is not None:
            metadata["notes"] = notes
        return metadata
