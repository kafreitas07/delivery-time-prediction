"""Treina e compara modelos de previsão do tempo de entrega.

Rodar da raiz do projeto: python -m src.train
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.data_prep import FEATURES, PROCESSED_PATH, TARGET

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"

RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def get_models() -> dict:
    return {
        "Baseline (média)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE),
    }


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def feature_importance(pipeline: Pipeline, top: int = 10) -> pd.Series:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:  # modelo linear: tamanho do coeficiente
        values = np.abs(model.coef_)
    return pd.Series(values, index=names).sort_values(ascending=False).head(top)


def main():
    df = pd.read_csv(PROCESSED_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape}  Teste: {X_test.shape}\n")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    holdout_rows, cv_rows, fitted = [], [], {}
    for name, model in get_models().items():
        print(f"Treinando {name}...")
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])

        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_rows.append({
            "Modelo": name,
            "MAE": -scores["test_MAE"].mean(),
            "MAE std": scores["test_MAE"].std(),
            "RMSE": -scores["test_RMSE"].mean(),
            "R2": scores["test_R2"].mean(),
            "R2 std": scores["test_R2"].std(),
        })

        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        train_m = metrics(y_train, pipe.predict(X_train))
        test_m = metrics(y_test, pipe.predict(X_test))
        holdout_rows.append({
            "Modelo": name,
            **{f"{k} treino": v for k, v in train_m.items()},
            **{f"{k} teste": v for k, v in test_m.items()},
        })

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    holdout = pd.DataFrame(holdout_rows).set_index("Modelo")
    print("\n=== Treino x teste ===")
    print(holdout.round(3))

    cv_table = pd.DataFrame(cv_rows).set_index("Modelo")
    print("\n=== Validação cruzada (5 folds, só no treino) ===")
    print(cv_table.round(3))

    # Melhor modelo escolhido pelo MAE da validação cruzada (o teste fica só para conferir)
    best_name = cv_table["MAE"].idxmin()
    best = fitted[best_name]
    print(f"\nMelhor modelo (menor MAE na validação cruzada): {best_name}")
    print(f"MAE no teste: {holdout.loc[best_name, 'MAE teste']:.2f} min")

    print(f"\n=== 10 variáveis mais importantes ({best_name}) ===")
    print(feature_importance(best).round(4).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"\nPipeline salvo em {MODEL_PATH}")


if __name__ == "__main__":
    main()
