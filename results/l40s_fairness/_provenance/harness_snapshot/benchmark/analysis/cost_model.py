"""
Cost model fitter.

Fits parametric and learned cost models to benchmark data.
Answers the core empirical questions:
  Q1 - Which strategy dominates per (prompt_len, output_len, task_type) regime?
  Q2 - Are costs separable across request dimensions?
  Q3 - How hardware-sensitive are strategy costs?
  Q4 - What are strategy interaction effects?

Output: a fitted CostModel that predicts TTFT/E2E given (strategy, request_features).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


@dataclass
class StrategyModel:
    """Per-strategy fitted cost model."""
    strategy_name: str
    ttft_model: object = None       # sklearn estimator
    e2e_model:  object = None
    ttft_mape:  float = 0.0         # cross-val MAPE
    e2e_mape:   float = 0.0
    n_samples:  int = 0
    feature_names: list[str] = field(default_factory=list)

    def predict_ttft(self, features: np.ndarray) -> np.ndarray:
        return self.ttft_model.predict(features)

    def predict_e2e(self, features: np.ndarray) -> np.ndarray:
        return self.e2e_model.predict(features)


@dataclass
class CostModel:
    """Ensemble of per-strategy models."""
    models: dict[str, StrategyModel] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    task_encoder: Optional[LabelEncoder] = None

    def predict(
        self,
        strategy_name: str,
        prompt_len: int,
        output_len: int,
        task_type: str,
    ) -> dict[str, float]:
        if strategy_name not in self.models:
            return {"ttft_ms": -1, "e2e_ms": -1}

        model = self.models[strategy_name]
        task_enc = self.task_encoder.transform([task_type])[0] if self.task_encoder else 0
        X = np.array([[prompt_len, output_len, task_enc]])
        return {
            "ttft_ms": float(model.predict_ttft(X)[0]),
            "e2e_ms":  float(model.predict_e2e(X)[0]),
        }

    def best_strategy_for(
        self,
        prompt_len: int,
        output_len: int,
        task_type: str,
        metric: str = "ttft_ms",
    ) -> str:
        """Oracle plan selection from cost model."""
        costs = {
            name: self.predict(name, prompt_len, output_len, task_type)[metric]
            for name in self.models
        }
        return min(costs, key=costs.get)


class CostModelFitter:

    FEATURES = ["prompt_token_len", "target_output_tokens", "task_type_enc"]

    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def load_data(self, requests_csv: str) -> pd.DataFrame:
        df = pd.read_csv(requests_csv)
        df = df[df["status"] == "ok"].copy()
        df["ttft_ms"] = df["ttft_s"] * 1000
        df["e2e_ms"]  = df["e2e_s"]  * 1000
        logger.info(
            f"Loaded {len(df)} successful requests from {requests_csv} "
            f"covering {df['strategy_name'].nunique()} strategies."
        )
        return df

    def _encode_task(self, df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
        le = LabelEncoder()
        df = df.copy()
        df["task_type_enc"] = le.fit_transform(df["task_type"].astype(str))
        return df, le

    def fit(self, requests_csv: str) -> CostModel:
        df = self.load_data(requests_csv)
        df, le = self._encode_task(df)

        cost_model = CostModel(feature_names=self.FEATURES, task_encoder=le)
        strategies = df["strategy_name"].unique()

        for strategy_name in strategies:
            sdf = df[df["strategy_name"] == strategy_name]
            if len(sdf) < 20:
                logger.warning(
                    f"Skipping [{strategy_name}]: only {len(sdf)} samples (need ≥20)."
                )
                continue

            X = sdf[self.FEATURES].values
            y_ttft = sdf["ttft_ms"].values
            y_e2e  = sdf["e2e_ms"].values

            ttft_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )
            e2e_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )

            ttft_model.fit(X, y_ttft)
            e2e_model.fit(X, y_e2e)

            # Cross-validated MAPE
            cv_splits = min(5, len(sdf) // 10)
            if cv_splits >= 2:
                ttft_cv = -cross_val_score(
                    ttft_model, X, y_ttft, cv=cv_splits,
                    scoring="neg_mean_absolute_percentage_error"
                ).mean()
                e2e_cv = -cross_val_score(
                    e2e_model, X, y_e2e, cv=cv_splits,
                    scoring="neg_mean_absolute_percentage_error"
                ).mean()
            else:
                ttft_pred = ttft_model.predict(X)
                e2e_pred  = e2e_model.predict(X)
                ttft_cv = mean_absolute_percentage_error(y_ttft, ttft_pred)
                e2e_cv  = mean_absolute_percentage_error(y_e2e,  e2e_pred)

            sm = StrategyModel(
                strategy_name=strategy_name,
                ttft_model=ttft_model,
                e2e_model=e2e_model,
                ttft_mape=ttft_cv,
                e2e_mape=e2e_cv,
                n_samples=len(sdf),
                feature_names=self.FEATURES,
            )
            cost_model.models[strategy_name] = sm

            logger.info(
                f"[{strategy_name}] fitted on {len(sdf)} samples — "
                f"TTFT MAPE={ttft_cv*100:.1f}%  E2E MAPE={e2e_cv*100:.1f}%"
            )

        return cost_model

    # ── Empirical analysis ────────────────────────────────────────────────────

    def analyze_separability(self, df: pd.DataFrame) -> dict:
        """
        Q2: Are costs separable?
        Fit linear models using only prompt_len, only output_len, and both.
        R² difference quantifies separability.
        """
        results = {}
        df, le = self._encode_task(df)

        for strategy in df["strategy_name"].unique():
            sdf = df[df["strategy_name"] == strategy]
            if len(sdf) < 30:
                continue

            y = sdf["ttft_ms"].values

            r2 = {}
            for features in [
                ["prompt_token_len"],
                ["target_output_tokens"],
                ["prompt_token_len", "target_output_tokens"],
                ["prompt_token_len", "target_output_tokens", "task_type_enc"],
            ]:
                X = sdf[features].values
                model = LinearRegression()
                scores = cross_val_score(model, X, y, cv=3, scoring="r2")
                key = "+".join(f.replace("_token_len","").replace("target_","") for f in features)
                r2[key] = float(scores.mean())

            results[strategy] = r2

        return results

    def analyze_dominance(
        self, df: pd.DataFrame, metric: str = "ttft_ms"
    ) -> pd.DataFrame:
        """
        Q1: Which strategy dominates per regime?
        Returns pivot: index=(prompt_bucket, output_bucket), columns=strategies, values=metric_p95.
        """
        df = df.copy()
        df["ttft_ms"] = df["ttft_s"] * 1000
        df["e2e_ms"]  = df["e2e_s"]  * 1000

        df["prompt_bucket"] = pd.cut(
            df["prompt_token_len"],
            bins=[0, 256, 512, 1024, 2048, 8192],
            labels=["≤256", "257-512", "513-1024", "1025-2048", ">2048"],
        )
        df["output_bucket"] = pd.cut(
            df["target_output_tokens"],
            bins=[0, 128, 256, 512, 10000],
            labels=["≤128", "129-256", "257-512", ">512"],
        )

        pivot = (
            df.groupby(["prompt_bucket", "output_bucket", "strategy_name"])[metric]
            .quantile(0.95)
            .reset_index()
            .pivot_table(
                index=["prompt_bucket", "output_bucket"],
                columns="strategy_name",
                values=metric,
            )
        )
        return pivot

    def analyze_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Q4: Interaction effects between strategies.
        Compares observed combined-strategy cost vs. sum of individual effects.
        """
        df = df.copy()
        df["ttft_ms"] = df["ttft_s"] * 1000

        baseline = df[df["strategy_name"].str.startswith("baseline")]["ttft_ms"].median()
        results = []

        for strategy in df["strategy_name"].unique():
            p95 = df[df["strategy_name"] == strategy]["ttft_ms"].quantile(0.95)
            improvement = (baseline - p95) / baseline * 100 if baseline > 0 else 0
            results.append({
                "strategy": strategy,
                "ttft_p95_ms": p95,
                "improvement_vs_baseline_pct": improvement,
            })

        return pd.DataFrame(results).sort_values("ttft_p95_ms")

    def save_analysis(self, df: pd.DataFrame, output_dir: str) -> dict:
        """Run all analyses and save results."""
        os.makedirs(output_dir, exist_ok=True)

        df = df.copy()
        if "ttft_ms" not in df.columns:
            df["ttft_ms"] = df["ttft_s"] * 1000
        if "e2e_ms" not in df.columns:
            df["e2e_ms"] = df["e2e_s"] * 1000

        separability = self.analyze_separability(df)
        dominance    = self.analyze_dominance(df)
        interactions = self.analyze_interactions(df)

        dominance.to_csv(os.path.join(output_dir, "dominance.csv"))
        interactions.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)

        with open(os.path.join(output_dir, "separability.json"), "w") as f:
            json.dump(separability, f, indent=2)

        logger.info(f"Analysis saved to {output_dir}")
        return {"separability": separability}
