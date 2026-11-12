import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_ORDER = {
    "restaurant_latitude": 22.745049,
    "restaurant_longitude": 75.892471,
    "delivery_latitude": 22.765049,
    "delivery_longitude": 75.912471,
    "delivery_person_age": 37,
    "delivery_person_rating": 4.9,
    "weather": "Sunny",
    "traffic": "High",
    "vehicle_condition": 2,
    "vehicle_type": "motorcycle",
    "order_type": "Snack",
    "multiple_deliveries": 0,
    "festival": False,
    "city": "Urban",
    "order_date": "2022-03-19",
    "order_time": "11:30:00",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # o "with" dispara o lifespan, que carrega o modelo
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_order(client):
    response = client.post("/predict", json=VALID_ORDER)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_minutes"}
    assert 5 <= body["predicted_minutes"] <= 60


@pytest.mark.parametrize(
    "field, value",
    [
        ("delivery_person_rating", 6.0),
        ("restaurant_latitude", 95.0),
        ("delivery_longitude", -200.0),
        ("weather", "Rainy"),
        ("traffic", "Heavy"),
    ],
)
def test_predict_invalid_order_returns_422(client, field, value):
    response = client.post("/predict", json={**VALID_ORDER, field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


def test_predict_distance_too_far_returns_422(client):
    # ~0,3 grau de latitude a mais ≈ 33 km do restaurante
    far = {**VALID_ORDER, "delivery_latitude": VALID_ORDER["restaurant_latitude"] + 0.3}
    response = client.post("/predict", json=far)
    assert response.status_code == 422
    assert "limite de 25 km" in response.json()["detail"][0]["msg"]


def test_predict_distance_within_limit(client):
    # ~0,2 grau ≈ 22 km: ainda dentro do limite
    near = {**VALID_ORDER, "delivery_latitude": VALID_ORDER["restaurant_latitude"] + 0.2}
    assert client.post("/predict", json=near).status_code == 200
