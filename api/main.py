"""API de previsão do tempo de entrega.

Rodar da raiz do projeto: uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from datetime import date, time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data_prep import DATE_FORMAT, FEATURES, TIME_FORMAT, build_features, haversine_km

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
# O treino só tem entregas de até ~21 km; acima disso a previsão não é confiável
MAX_DISTANCE_KM = 25

Latitude = Field(ge=-90, le=90)
Longitude = Field(ge=-180, le=180)


class Order(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    restaurant_latitude: float = Latitude
    restaurant_longitude: float = Longitude
    delivery_latitude: float = Latitude
    delivery_longitude: float = Longitude
    delivery_person_age: int = Field(ge=15, le=80)
    delivery_person_rating: float = Field(ge=1, le=5)
    weather: Literal["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]
    traffic: Literal["Low", "Medium", "High", "Jam"]
    vehicle_condition: int = Field(ge=0, le=3)
    vehicle_type: Literal["motorcycle", "scooter", "electric_scooter", "bicycle"]
    order_type: Literal["Snack", "Meal", "Drinks", "Buffet"]
    multiple_deliveries: int = Field(ge=0, le=3)
    festival: bool
    city: Literal["Urban", "Metropolitian", "Semi-Urban"]  # grafia igual à do dataset
    order_date: date
    order_time: time

    @model_validator(mode="after")
    def check_distance(self):
        distance = haversine_km(self.restaurant_latitude, self.restaurant_longitude,
                                self.delivery_latitude, self.delivery_longitude)
        if distance > MAX_DISTANCE_KM:
            raise ValueError(
                f"distância restaurante → entrega de {distance:.1f} km "
                f"passa do limite de {MAX_DISTANCE_KM} km"
            )
        return self


class Prediction(BaseModel):
    predicted_minutes: float


def order_to_raw(order: Order) -> pd.DataFrame:
    """Converte o pedido para as colunas do CSV bruto, para passar por build_features."""
    return pd.DataFrame([{
        "Restaurant_latitude": order.restaurant_latitude,
        "Restaurant_longitude": order.restaurant_longitude,
        "Delivery_location_latitude": order.delivery_latitude,
        "Delivery_location_longitude": order.delivery_longitude,
        "Delivery_person_Age": order.delivery_person_age,
        "Delivery_person_Ratings": order.delivery_person_rating,
        "Weatherconditions": order.weather,
        "Road_traffic_density": order.traffic,
        "Vehicle_condition": order.vehicle_condition,
        "Type_of_vehicle": order.vehicle_type,
        "Type_of_order": order.order_type,
        "multiple_deliveries": order.multiple_deliveries,
        "Festival": "Yes" if order.festival else "No",
        "City": order.city,
        "Order_Date": order.order_date.strftime(DATE_FORMAT),
        "Time_Orderd": order.order_time.strftime(TIME_FORMAT),
    }])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)  # carregado uma vez, na inicialização
    yield


app = FastAPI(title="Delivery Time Prediction", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(order: Order):
    features = build_features(order_to_raw(order))[FEATURES]
    minutes = float(app.state.model.predict(features)[0])
    return {"predicted_minutes": round(minutes, 1)}
