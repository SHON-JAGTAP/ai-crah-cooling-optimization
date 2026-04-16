"""
modules/preprocessor.py
-----------------------
Data preprocessing pipeline for CRAH sensor data.

Changes from v1:
  + airflow_dist_factor added to SCALE_COLS (4th scaled feature for ML)
  + PHYSICAL_COLS updated for compound condition support
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from config import SMOOTHING_WINDOW, Z_SCORE_THRESHOLD

CLIP_BOUNDS = {
    "rack_temp_c":          (0.0,  80.0),
    "airflow_cfm":          (100.0, 3000.0),
    "discharge_temp_c":     (5.0,  30.0),
    "it_load_kw":           (0.0,  20.0),
    "airflow_dist_factor":  (0.1,  5.0),
}


class Preprocessor:
    """
    Stateful preprocessor: fit on training data, transform on live data.
    Scales only physical value columns (not integer IDs or string aisle names).
    """

    PHYSICAL_COLS = ["airflow_cfm", "discharge_temp_c", "it_load_kw",
                     "airflow_dist_factor", "rack_temp_c"]

    SCALE_COLS    = ["airflow_cfm", "discharge_temp_c", "it_load_kw",
                     "airflow_dist_factor", "rack_temp_c"]

    def __init__(self):
        self._scaler  = MinMaxScaler()
        self._fitted  = False
        self._num_cols: list[str] = []

    # -- Helpers ---------------------------------------------------------------

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            print(f"  [Preprocessor] Dropped {dropped} duplicate rows.")
        return df

    @staticmethod
    def _clip_physical(df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in CLIP_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        before = len(df)
        outlier_cols = [c for c in Preprocessor.PHYSICAL_COLS if c in df.columns]
        if not outlier_cols:
            return df
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z_scores = np.abs(stats.zscore(df[outlier_cols].astype(float)))
        mask = (z_scores < Z_SCORE_THRESHOLD).all(axis=1)
        candidate = df[mask].reset_index(drop=True)
        if len(candidate) == 0:
            print("  [Preprocessor] Outlier removal would empty dataset -- skipping.")
            return df
        removed = before - len(candidate)
        if removed:
            print(f"  [Preprocessor] Removed {removed} outlier rows (Z>{Z_SCORE_THRESHOLD}).")
        return candidate

    @staticmethod
    def _smooth(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .rolling(window=SMOOTHING_WINDOW, min_periods=1, center=True)
                    .mean()
                    .round(4)
                )
        return df

    def _identify_scale_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.SCALE_COLS if c in df.columns]

    # -- Public API -----------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[Preprocessor] Starting fit_transform pipeline ...")
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._clip_physical(df)

        self._num_cols = self._identify_scale_cols(df)
        df = self._remove_outliers(df, self._num_cols)
        df = self._smooth(df, self._num_cols)

        df[self._num_cols] = self._scaler.fit_transform(df[self._num_cols])
        self._fitted = True
        print(f"  [Preprocessor] Complete. Shape: {df.shape}, scaled cols: {self._num_cols}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit_transform'd first.")
        df = df.copy()
        df = self._clip_physical(df)
        num_cols_present = [c for c in self._num_cols if c in df.columns]
        full_input = df[num_cols_present].reindex(columns=self._num_cols, fill_value=0)
        scaled = self._scaler.transform(full_input)
        indices = [self._num_cols.index(c) for c in num_cols_present]
        df[num_cols_present] = scaled[:, indices]
        return df

    def inverse_transform_column(self, values: np.ndarray, column: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit_transform'd first.")
        idx = self._num_cols.index(column)
        dummy = np.zeros((len(values), len(self._num_cols)))
        dummy[:, idx] = values
        return self._scaler.inverse_transform(dummy)[:, idx]

    @property
    def feature_columns(self) -> list[str]:
        return self._num_cols
