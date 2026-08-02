import pytest
from app import app, load_and_prepare, FEATURES

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200

def test_forecast_endpoint_returns_200(client):
    response = client.get("/forecast")
    assert response.status_code == 200

def test_forecast_returns_json_list(client):
    response = client.get("/forecast")
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 12

def test_forecast_items_have_correct_keys(client):
    response = client.get("/forecast")
    data = response.get_json()
    for item in data:
        assert "date" in item
        assert "forecast" in item
        assert isinstance(item["forecast"], float)

def test_load_and_prepare_has_required_columns(client):
    df = load_and_prepare()
    for col in FEATURES:
        assert col in df.columns