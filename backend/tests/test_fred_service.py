import httpx
import pytest
from services.fred_service import FredApiError, FredService


def make_service(handler) -> FredService:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return FredService(api_key="test-key", client=client)


@pytest.mark.asyncio
async def test_get_category_series_returns_series_list():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/fred/category/series"
        assert request.url.params["category_id"] == "9"
        assert request.url.params["api_key"] == "test-key"
        return httpx.Response(200, json={"seriess": [{"id": "CPIAUCSL"}]})

    service = make_service(handler)
    result = await service.get_category_series(9)
    assert result == [{"id": "CPIAUCSL"}]


@pytest.mark.asyncio
async def test_get_category_children_returns_categories_list():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/fred/category/children"
        return httpx.Response(200, json={"categories": [{"id": 1, "name": "Sub"}]})

    service = make_service(handler)
    result = await service.get_category_children(22)
    assert result == [{"id": 1, "name": "Sub"}]


@pytest.mark.asyncio
async def test_get_series_observations_maps_date_and_value():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["series_id"] == "CPIAUCSL"
        assert "observation_start" in request.url.params
        return httpx.Response(
            200,
            json={"observations": [{"date": "2024-01-01", "value": "1.0", "extra": "ignored"}]},
        )

    service = make_service(handler)
    result = await service.get_series_observations("CPIAUCSL")
    assert result == [{"date": "2024-01-01", "value": "1.0"}]


@pytest.mark.asyncio
async def test_get_series_observations_honors_explicit_start():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["observation_start"] == "2020-01-01"
        return httpx.Response(200, json={"observations": []})

    service = make_service(handler)
    await service.get_series_observations("CPIAUCSL", observation_start="2020-01-01")


@pytest.mark.asyncio
async def test_get_series_observations_raises_on_bad_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error_message": "Bad Request"})

    service = make_service(handler)
    with pytest.raises(FredApiError) as excinfo:
        await service.get_series_observations("BAD_SERIES")
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_get_series_observations_raises_when_observations_key_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    service = make_service(handler)
    with pytest.raises(FredApiError):
        await service.get_series_observations("SOME_SERIES")


def test_get_series_metadata_includes_notes_when_present():
    raw = {
        "id": "CPIAUCSL",
        "title": "CPI",
        "frequency": "Monthly",
        "units": "Index",
        "seasonal_adjustment": "SA",
        "notes": "Some notes",
    }
    metadata = FredService.get_series_metadata(raw)
    assert metadata == {
        "id": "CPIAUCSL",
        "title": "CPI",
        "frequency": "Monthly",
        "units": "Index",
        "seasonal_adjustment": "SA",
        "notes": "Some notes",
    }


def test_get_series_metadata_omits_notes_when_absent():
    raw = {
        "id": "CPIAUCSL",
        "title": "CPI",
        "frequency": "Monthly",
        "units": "Index",
        "seasonal_adjustment": "SA",
    }
    metadata = FredService.get_series_metadata(raw)
    assert "notes" not in metadata
