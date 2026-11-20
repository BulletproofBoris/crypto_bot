#!/usr/bin/env python3
"""
_tools/order2_risk_predictor.py
=============================================================================
High-Speed Vectorized Predictor for all 11 Order-2 Risk & Microstructure Models.
Loads native binary .ubj models and evaluates a batch or single tick in microseconds.
=============================================================================
"""

from pathlib import Path
import json
import numpy as np
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
ORDER2_DIR = BASE_DIR / "models" / "prl" / "order2_risk"

TARGET_NAMES = [
    "adverse_selection_5s",
    "wave_exhaustion_10s",
    "ask_sweep_5s",
    "guard_sweep_12s",
    "guard_bleed_60s",
    "guard_crash_300s",
    "guard_exit_30s",
    "exec_fill_30s",
    "exec_frontrun_30s",
    "carousel_success",
    "carousel_capacity",
]


class Order2RiskPredictor:
    """Zero-overhead evaluator for all 11 Order-2 XGBoost models."""

    def __init__(self, models_dir: Path = ORDER2_DIR):
        self.models = {}
        for name in TARGET_NAMES:
            model_path = models_dir / f"{name}.ubj"
            if model_path.exists():
                bst = xgb.Booster()
                bst.load_model(str(model_path))
                self.models[name] = bst
            else:
                print(f"⚠️ Warning: Model {model_path} not found!")

    def predict_batch(self, X_38d: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluates batch of [N, 38] features -> returns dict of 11 arrays."""
        d = xgb.DMatrix(X_38d)
        res = {}
        for name, bst in self.models.items():
            res[name] = bst.predict(d)
        return res

    def predict_matrix_11d(self, X_38d: np.ndarray) -> np.ndarray:
        """Returns [N, 11] matrix with all risk signals aligned in column order."""
        d = xgb.DMatrix(X_38d)
        cols = [self.models[name].predict(d) for name in TARGET_NAMES if name in self.models]
        return np.column_stack(cols).astype(np.float32)


if __name__ == "__main__":
    import time
    print("Testing Order2RiskPredictor...")
    pred = Order2RiskPredictor()
    dummy_x = np.random.randn(1000, 38).astype(np.float32)
    t0 = time.time()
    out = pred.predict_matrix_11d(dummy_x)
    el = (time.time() - t0) * 1000
    print(f"✅ Successfully evaluated 1,000 ticks in {el:.2f} ms ({el/1000*1000:.1f} μs per tick)!")
    print(f"Output shape: {out.shape}")
