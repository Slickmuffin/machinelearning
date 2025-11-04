#!/usr/bin/env python3
"""
EnsemblePredictor - Load and use trained swing trading ensembles

Usage:
    # After downloading artifacts from GCS
    from ensemble_loader import EnsemblePredictor

    predictor = EnsemblePredictor("./artifacts/experiment-123/")
    probs = predictor.predict_proba(df_features)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load as joblib_load
import xgboost as xgb
from scipy.special import expit, logit


class EnsemblePredictor:
    """Load and use trained ensemble for predictions"""

    def __init__(self, artifact_dir: str):
        """
        Load ensemble from artifact directory

        Args:
            artifact_dir: Path to directory containing manifest.json, models, calibrators
        """
        self.dir = Path(artifact_dir)

        # Load manifest
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in {artifact_dir}")

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.feature_cols = self.manifest["feature_cols"]
        self.combiner = self.manifest.get("combiner", "logit_avg")
        self.models = []

        # Load each model + calibrator
        for model_spec in self.manifest["models"]:
            booster = xgb.Booster()
            booster.load_model(str(self.dir / model_spec["booster_file"]))

            calibrator = None
            if model_spec.get("calibrator_file"):
                calibrator = joblib_load(self.dir / model_spec["calibrator_file"])

            self.models.append({
                "booster": booster,
                "calibrator": calibrator,
                "feature_cols": model_spec["feature_cols"],
                "label": model_spec["label"],
                "val_auc": model_spec.get("val_auc", 0.0),
                "best_iter": model_spec.get("best_iter", 0),
                "seed": model_spec.get("seed", 0)
            })

        print(f"✅ Loaded ensemble: {len(self.models)} models, {len(self.feature_cols)} features")
        print(f"   Combiner: {self.combiner}")
        print(f"   Avg AUC: {np.mean([m['val_auc'] for m in self.models]):.3f}")

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions

        Args:
            df_features: DataFrame with feature columns (must contain all self.feature_cols)

        Returns:
            Array of probabilities (one per row)
        """
        # Validate features
        missing_cols = [col for col in self.feature_cols if col not in df_features.columns]
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols[:5]}...")

        # Collect predictions from each model
        probs = []
        for model_info in self.models:
            # Extract features this model uses
            model_features = model_info['feature_cols']
            X_model = df_features[model_features].astype("float32").values

            # Predict with booster
            dmat = xgb.DMatrix(X_model)
            p = model_info["booster"].predict(dmat)

            # Apply calibration if available
            if model_info["calibrator"] is not None:
                # Calibrator expects probabilities as input
                p = model_info["calibrator"].predict_proba(X_model)[:, 1]

            probs.append(p)

        # Stack probabilities: shape [n_models, n_samples]
        P = np.vstack(probs)

        # Combine according to strategy
        if self.combiner == "avg":
            return P.mean(axis=0)
        elif self.combiner == "logit_avg":
            # Average in logit space (more stable than averaging probabilities)
            eps = 1e-6
            L = logit(np.clip(P, eps, 1 - eps))
            return expit(L.mean(axis=0))
        else:
            raise ValueError(f"Unknown combiner: {self.combiner}")

    def predict_ensemble(self, X: np.ndarray) -> tuple:
        """
        Get ensemble prediction with uncertainty (matches training API)

        Args:
            X: Feature array [n_samples, n_features] or [n_features] for single sample

        Returns:
            (p_ens, z_std, individual_probs)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = []
        for model_info in self.models:
            feature_indices = [self.feature_cols.index(f) for f in model_info['feature_cols'] if f in self.feature_cols]
            if len(feature_indices) > 0:
                X_model = X[:, feature_indices]
                dmat = xgb.DMatrix(X_model)
                prob = model_info['booster'].predict(dmat)[0]
                probs.append(prob)

        if not probs:
            return 0.5, 1.0, [0.5] * len(self.models)

        # Logit averaging
        z = np.array([logit(np.clip(p, 1e-6, 1-1e-6)) for p in probs])
        p_ens = float(expit(z.mean()))
        z_std = float(z.std())

        return p_ens, z_std, probs

    def info(self) -> dict:
        """Get ensemble info"""
        return {
            "n_models": len(self.models),
            "combiner": self.combiner,
            "n_features": len(self.feature_cols),
            "avg_auc": np.mean([m['val_auc'] for m in self.models]),
            "avg_trees": np.mean([m['best_iter'] for m in self.models]),
            "labels": [m['label'] for m in self.models]
        }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect ensemble")
    parser.add_argument("artifact_dir", help="Path to artifact directory")
    parser.add_argument("--test_csv", help="Optional CSV to test predictions")
    args = parser.parse_args()

    # Load ensemble
    predictor = EnsemblePredictor(args.artifact_dir)

    # Show info
    info = predictor.info()
    print(f"\n📊 Ensemble Info:")
    for k, v in info.items():
        print(f"   {k}: {v}")

    # Test predictions if CSV provided
    if args.test_csv:
        df = pd.read_csv(args.test_csv)
        probs = predictor.predict_proba(df)
        print(f"\n🎯 Test Predictions:")
        print(f"   Shape: {probs.shape}")
        print(f"   Mean prob: {probs.mean():.3f}")
        print(f"   Std prob: {probs.std():.3f}")
        print(f"   Min/Max: {probs.min():.3f} / {probs.max():.3f}")
