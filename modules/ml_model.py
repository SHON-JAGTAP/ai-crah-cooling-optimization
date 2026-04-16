"""
modules/ml_model.py
-------------------
Ensemble RF + GB model: T_rack = f(F_CRAH, T_discharge, L_IT, A_airflow)

Changes from v1:
  + airflow_dist_factor added as 4th feature (A_airflow from document)
  + Full 4-feature model: T_rack = f(airflow_cfm, discharge_temp_c, it_load_kw, airflow_dist_factor)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    ML_MAX_DEPTH, ML_N_ESTIMATORS, ML_RANDOM_STATE, ML_TEST_SIZE,
    MODEL_SAVE_PATH,
)

# 4-feature model as documented:
# T_rack = f(F_CRAH=airflow_cfm, T_discharge=discharge_temp_c,
#             L_IT=it_load_kw,   A_airflow=airflow_dist_factor)
FEATURES = ["airflow_cfm", "discharge_temp_c", "it_load_kw", "airflow_dist_factor"]
TARGET   = "rack_temp_c"


class TemperaturePredictor:
    """RF + GB ensemble for rack temperature prediction with 4 input features."""

    def __init__(self):
        self._rf = RandomForestRegressor(
            n_estimators=ML_N_ESTIMATORS,
            max_depth=ML_MAX_DEPTH,
            random_state=ML_RANDOM_STATE,
            n_jobs=-1,
        )
        self._gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=ML_RANDOM_STATE,
            learning_rate=0.05,
        )
        self._preprocessor = None
        self._trained = False
        self.metrics: dict = {}

    # -- Training --------------------------------------------------------------

    def train(self, scaled_df: pd.DataFrame, preprocessor) -> dict:
        self._preprocessor = preprocessor
        feat_cols = [f for f in FEATURES if f in scaled_df.columns]
        if TARGET not in scaled_df.columns:
            raise ValueError(f"Target '{TARGET}' not in dataframe.")

        X = scaled_df[feat_cols].values
        y = scaled_df[TARGET].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ML_TEST_SIZE, random_state=ML_RANDOM_STATE
        )

        print("[ML Model] Training RandomForest (4 features: F_CRAH, T_d, L_IT, A_air) ...")
        self._rf.fit(X_train, y_train)
        print("[ML Model] Training GradientBoosting ...")
        self._gb.fit(X_train, y_train)

        rf_pred  = self._rf.predict(X_test)
        gb_pred  = self._gb.predict(X_test)
        ens_pred = 0.6 * rf_pred + 0.4 * gb_pred

        rf_c   = preprocessor.inverse_transform_column(rf_pred,  TARGET)
        gb_c   = preprocessor.inverse_transform_column(gb_pred,  TARGET)
        ens_c  = preprocessor.inverse_transform_column(ens_pred, TARGET)
        y_c    = preprocessor.inverse_transform_column(y_test,   TARGET)

        self.metrics = {
            "rf_mae_c":       round(mean_absolute_error(y_c, rf_c),  3),
            "gb_mae_c":       round(mean_absolute_error(y_c, gb_c),  3),
            "ensemble_mae_c": round(mean_absolute_error(y_c, ens_c), 3),
            "rf_r2":          round(r2_score(y_test, rf_pred),  4),
            "gb_r2":          round(r2_score(y_test, gb_pred),  4),
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "features":       feat_cols,
            "feature_importances": dict(
                zip(feat_cols, [round(v, 4) for v in self._rf.feature_importances_])
            ),
        }
        self._trained = True
        print(f"  RF  MAE: {self.metrics['rf_mae_c']:.3f} deg-C  R2: {self.metrics['rf_r2']:.4f}")
        print(f"  GB  MAE: {self.metrics['gb_mae_c']:.3f} deg-C  R2: {self.metrics['gb_r2']:.4f}")
        print(f"  Ens MAE: {self.metrics['ensemble_mae_c']:.3f} deg-C")
        return self.metrics

    # -- Inference -------------------------------------------------------------

    def predict(self, scaled_df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained.")
        feat_cols = [f for f in FEATURES if f in scaled_df.columns]
        X = scaled_df[feat_cols].values
        rf_pred  = self._rf.predict(X)
        gb_pred  = self._gb.predict(X)
        ens_pred = 0.6 * rf_pred + 0.4 * gb_pred
        return self._preprocessor.inverse_transform_column(ens_pred, TARGET)

    def predict_single(
        self,
        airflow_cfm: float,
        discharge_temp_c: float,
        it_load_kw: float,
        airflow_dist_factor: float = 1.0,
    ) -> float:
        """Convenience: predict for raw (un-normalised) feature values."""
        row = pd.DataFrame([{
            "airflow_cfm":         airflow_cfm,
            "discharge_temp_c":    discharge_temp_c,
            "it_load_kw":          it_load_kw,
            "airflow_dist_factor": airflow_dist_factor,
        }])
        scaled_row = self._preprocessor.transform(row)
        return float(self.predict(scaled_row)[0])

    # -- Persistence -----------------------------------------------------------

    def save(self, path: str = MODEL_SAVE_PATH) -> None:
        joblib.dump({"rf": self._rf, "gb": self._gb, "metrics": self.metrics}, path)
        print(f"[ML Model] Saved to {path}")

    def load(self, path: str = MODEL_SAVE_PATH, preprocessor=None) -> None:
        obj = joblib.load(path)
        self._rf, self._gb  = obj["rf"], obj["gb"]
        self.metrics        = obj.get("metrics", {})
        self._preprocessor  = preprocessor
        self._trained       = True
        print(f"[ML Model] Loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._trained
