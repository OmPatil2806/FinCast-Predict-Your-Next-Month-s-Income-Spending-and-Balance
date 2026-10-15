"""The forecasting model as the app uses it - no MLflow, no training code.

    BASELINES          the three naive forecasts (step 6.9), computed from the feature rows
    ForecastModel      one LightGBM model (a family, a target mode, a quantile) plus what it needs
    widen              stretch a forecast range around its median by a factor (step 7.4)
    apply_champions    champion point per series + LightGBM range (step 7.8)
    ChampionForecaster the exported bundle per family (step 7.9): q10 / median / q90 models, the
                       champion per series and the range width; .predict(feature rows) -> lower,
                       predicted, upper. Saved with joblib to models/forecaster_<family>.joblib.
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

LIGHTGBM = "lightgbm"

BASELINES = {
    "last_month": lambda t: t["lag_1"],
    "same_month_last_year": lambda t: t["lag_12"].fillna(t["lag_1"]),
    "mean_3": lambda t: t["roll_mean_3"].fillna(t["lag_1"]),
}

DEFAULT_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": 4,
}


def scale_of(rows: pd.DataFrame) -> pd.Series:
    """The series' recent level, used by the relative mode."""
    return rows["roll_mean_6"].fillna(rows["roll_mean_3"]).fillna(rows["lag_1"]).fillna(0.0).clip(lower=0.0)


def to_model_target(rows: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "raw":
        return rows["target"]
    return np.log1p(rows["target"].clip(lower=0)) - np.log1p(scale_of(rows))


def from_model_output(rows: pd.DataFrame, output: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.clip(output, 0, None)
    return np.clip(np.expm1(output + np.log1p(scale_of(rows).to_numpy())), 0, None)


def model_inputs(rows: pd.DataFrame, feature_columns: list[str], series_levels: list[str]) -> pd.DataFrame:
    X = rows[feature_columns].copy()
    if len(series_levels) > 1:
        X["series"] = pd.Categorical(rows["series"], categories=series_levels)
    return X


@dataclass
class ForecastModel:
    """A fitted LightGBM model plus everything needed to use it on new rows."""
    family: str
    mode: str
    quantile: float
    feature_columns: list[str]
    series_levels: list[str]
    params: dict = field(default_factory=dict)
    booster: lgb.LGBMRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> "ForecastModel":
        objective = {"objective": "l1"} if self.quantile == 0.5 else {"objective": "quantile", "alpha": self.quantile}
        self.booster = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **self.params, **objective})
        self.booster.fit(model_inputs(rows, self.feature_columns, self.series_levels), to_model_target(rows, self.mode))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        output = self.booster.predict(model_inputs(rows, self.feature_columns, self.series_levels))
        return from_model_output(rows, output, self.mode)


def widen(predictions: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale the distance from the median to each bound by `factor` (>1 wider, <1 narrower)."""
    out = predictions.copy()
    out["lower"] = np.clip(out["predicted"] - factor * (out["predicted"] - predictions["lower"]), 0, None)
    out["upper"] = out["predicted"] + factor * (predictions["upper"] - out["predicted"])
    return out


def apply_champions(rows: pd.DataFrame, lower: np.ndarray, median: np.ndarray, upper: np.ndarray,
                    champions: dict, factor: float) -> pd.DataFrame:
    """Final forecast: champion point per series, LightGBM range (widened), stretched to hold the point.

    rows: feature rows (need series + the lag columns the baselines use). champions: series -> name;
    a series without an entry uses LightGBM.
    """
    point = np.asarray(median, dtype=float).copy()
    names = rows["series"].map(champions).fillna(LIGHTGBM).to_numpy()
    for name, forecast in BASELINES.items():
        mask = names == name
        if mask.any():
            point[mask] = forecast(rows[mask]).to_numpy()
    band = widen(pd.DataFrame({"lower": lower, "predicted": median, "upper": upper}), factor)
    point = np.clip(point, 0, None)
    return pd.DataFrame({"champion": names, "lower": np.minimum(band["lower"].to_numpy(), point),
                         "predicted": point, "upper": np.maximum(band["upper"].to_numpy(), point)},
                        index=rows.index)


@dataclass
class ChampionForecaster:
    """Everything one family needs to forecast: the exported, ready-to-use champion (step 7.9)."""
    family: str
    mode: str
    models: dict                       # quantile -> fitted ForecastModel (0.1, 0.5, 0.9)
    champions: dict                    # series -> "lightgbm" or a baseline name
    width_factor: float = 1.0
    version: int | None = None

    @property
    def feature_columns(self) -> list[str]:
        return self.models[0.5].feature_columns

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Feature rows (from build_features) -> champion, lower, predicted, upper (in euro), same index."""
        values = np.column_stack([self.models[q].predict(rows) for q in sorted(self.models)])
        values.sort(axis=1)                                   # quantile models trained apart can cross
        return apply_champions(rows, values[:, 0], values[:, 1], values[:, 2], self.champions, self.width_factor)
