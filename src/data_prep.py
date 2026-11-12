"""Limpeza do dataset de tempo de entrega."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "train.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "clean.csv"

TARGET = "Time_taken(min)"
MAX_DISTANCE_KM = 50
COORD_COLS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]
DROP_COLS = ["ID", "Delivery_person_ID", "Time_Order_picked", "Order_Date", "Time_Orderd"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos (vetorizada)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Espaços sobrando e "NaN" em texto -> np.nan
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip().replace({"NaN": np.nan, "conditions NaN": np.nan})

    # 2. Prefixos e alvo
    df["Weatherconditions"] = df["Weatherconditions"].str.removeprefix("conditions ")
    df[TARGET] = df[TARGET].str.removeprefix("(min) ").astype(int)

    # 3. Colunas numéricas lidas como texto
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"])
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"])
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"])

    # Avaliações fora da escala (acima de 5) viram ausentes
    df.loc[df["Delivery_person_Ratings"] > 5, "Delivery_person_Ratings"] = np.nan

    # 4. Distância restaurante -> entrega
    # Algumas coordenadas vêm com o sinal trocado
    df[COORD_COLS] = df[COORD_COLS].abs()
    df["distance_km"] = haversine_km(*(df[c] for c in COORD_COLS))
    zero_coords = (df[COORD_COLS] == 0).any(axis=1)
    df = df[~zero_coords & (df["distance_km"] <= MAX_DISTANCE_KM)]
    df = df.drop(columns=COORD_COLS)

    # 5. Data e hora do pedido
    order_date = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    order_time = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    df["order_hour"] = order_time.dt.hour  # NaN quando Time_Orderd é ausente
    df["day_of_week"] = order_date.dt.day_of_week  # 0 = segunda

    # 6. Colunas que não entram no modelo
    df = df.drop(columns=DROP_COLS)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    clean = clean_data(raw)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"Linhas removidas: {len(raw) - len(clean)}")
    print("shape:", clean.shape)
    print("\nisna().sum():\n", clean.isna().sum(), sep="")
    print("\ndescribe():\n", clean.describe(), sep="")
    print(f"\nSalvo em {PROCESSED_PATH}")
