"""
Report generator.

Produces paper-quality figures and LaTeX tables from benchmark results.
Covers the five research questions defined in the benchmark design.
"""

import json
import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = "tab10"
FIG_DPI = 300
FONT_SIZE = 10
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": FIG_DPI,
})


class ReportGenerator:

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.fig_dir = os.path.join(run_dir, "figures")
        self.table_dir = os.path.join(run_dir, "tables")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None
        self._summaries: Optional[list[dict]] = None

    def _load(self):
        req_csv  = os.path.join(self.run_dir, "requests.csv")
        sum_json = os.path.join(self.run_dir, "summary.json")

        if os.path.exists(req_csv):
            self._df = pd.read_csv(req_csv)
            self._df["ttft_ms"] = self._df["ttft_s"] * 1000
            self._df["e2e_ms"]  = self._df["e2e_s"]  * 1000
            self._df = self._df[self._df["status"] == "ok"]

        if os.path.exists(sum_json):
            with open(sum_json) as f:
                self._summaries = json.load(f)

    def generate_all(self) -> list[str]:
        """Generate all figures and tables. Returns list of output paths."""
        self._load()
        if self._df is None or self._df.empty:
            logger.warning("No data to report.")
            return []

        paths = []
        paths += self.fig_ttft_cdf_by_strategy()
        paths += self.fig_dominance_heatmap()
        paths += self.fig_strategy_comparison_boxplot()
        paths += self.fig_kv_cache_vs_ttft()
        paths += self.fig_tps_by_output_length()
        paths += self.table_summary_latex()
        paths += self.table_efficiency_latex()
        paths += self.table_interaction_effects()
        logger.info(f"Report generated: {len(paths)} artifacts.")
        return paths

    # ── Figure 1: TTFT CDF by strategy ───────────────────────────────────────
    def fig_ttft_cdf_by_strategy(self) -> list[str]:
        """CDF of TTFT across strategies — shows tail behavior."""
        df = self._df
        strategies = df["strategy_name"].unique()
        palette = sns.color_palette(PALETTE, len(strategies))

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, strat in enumerate(sorted(strategies)):
            vals = np.sort(df[df["strategy_name"] == strat]["ttft_ms"].values)
            cdf  = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=strat, color=palette[i], linewidth=1.5)

        ax.axvline(x=df.groupby("strategy_name")["ttft_ms"].quantile(0.95).min(),
                   linestyle="--", color="gray", linewidth=0.8, label="best P95")
        ax.set_xlabel("TTFT (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("TTFT CDF by Execution Strategy")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "ttft_cdf.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return [path]

    # ── Figure 2: Dominance heatmap ───────────────────────────────────────────
    def fig_dominance_heatmap(self) -> list[str]:
        """
        Heatmap: for each (prompt_bucket, output_bucket), which strategy has lowest TTFT P95?
        Directly answers RQ1.
        """
        df = self._df.copy()
        df["prompt_bucket"] = pd.cut(
            df["prompt_token_len"],
            bins=[0, 256, 512, 1024, 2048, 99999],
            labels=["≤256", "257–512", "513–1024", "1025–2048", ">2048"],
        )
        df["output_bucket"] = pd.cut(
            df["target_output_tokens"],
            bins=[0, 128, 256, 512, 99999],
            labels=["≤128", "129–256", "257–512", ">512"],
        )

        pivot = (
            df.groupby(["prompt_bucket", "output_bucket", "strategy_name"])["ttft_ms"]
            .quantile(0.95)
            .reset_index()
        )
        best = pivot.loc[pivot.groupby(["prompt_bucket", "output_bucket"])["ttft_ms"].idxmin()]

        strategies = sorted(df["strategy_name"].unique())
        strat_to_int = {s: i for i, s in enumerate(strategies)}
        best["strat_int"] = best["strategy_name"].map(strat_to_int)

        heat_df = best.pivot(
            index="prompt_bucket", columns="output_bucket", values="strat_int"
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = matplotlib.colormaps["tab10"].resampled(len(strategies))
        im = ax.imshow(heat_df.values, cmap=cmap, aspect="auto",
                       vmin=-0.5, vmax=len(strategies) - 0.5)

        ax.set_xticks(range(len(heat_df.columns)))
        ax.set_xticklabels(heat_df.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(heat_df.index)))
        ax.set_yticklabels(heat_df.index, fontsize=8)
        ax.set_xlabel("Output Length Bucket (tokens)")
        ax.set_ylabel("Prompt Length Bucket (tokens)")
        ax.set_title("Best Strategy per (Prompt, Output) Regime — TTFT P95")

        cbar = plt.colorbar(im, ax=ax, ticks=range(len(strategies)))
        cbar.ax.set_yticklabels(strategies, fontsize=6)

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "dominance_heatmap.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return [path]

    # ── Figure 3: Strategy comparison boxplot ─────────────────────────────────
    def fig_strategy_comparison_boxplot(self) -> list[str]:
        """Side-by-side boxplots of TTFT and E2E P95 per strategy."""
        df = self._df
        paths = []

        for metric, label in [("ttft_ms", "TTFT (ms)"), ("e2e_ms", "E2E Latency (ms)")]:
            fig, ax = plt.subplots(figsize=(9, 4))
            order = (
                df.groupby("strategy_name")[metric].quantile(0.95)
                .sort_values().index.tolist()
            )
            sns.boxplot(
                data=df, x="strategy_name", y=metric, order=order,
                hue="strategy_name", palette=PALETTE, fliersize=2, ax=ax,
                showfliers=False, legend=False,
            )
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.set_title(f"{label} Distribution by Execution Strategy")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
            plt.tight_layout()

            fname = f"{metric.replace('_ms','')}_boxplot.pdf"
            path  = os.path.join(self.fig_dir, fname)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
            logger.info(f"Saved {path}")

        return paths

    # ── Figure 4: KV cache utilization vs TTFT ────────────────────────────────
    def fig_kv_cache_vs_ttft(self) -> list[str]:
        """Scatter: KV cache utilization (%) vs TTFT P95 per strategy."""
        df = self._df[self._df["kv_cache_usage_pct"] > 0].copy()
        if df.empty:
            return []

        strategies = df["strategy_name"].unique()
        palette = sns.color_palette(PALETTE, len(strategies))

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, strat in enumerate(sorted(strategies)):
            sdf = df[df["strategy_name"] == strat]
            ax.scatter(
                sdf["kv_cache_usage_pct"],
                sdf["ttft_ms"],
                label=strat, alpha=0.4, s=8, color=palette[i],
            )

        ax.set_xlabel("KV Cache Utilization (%)")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title("KV Cache Pressure vs. TTFT")
        ax.legend(fontsize=7)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "kv_cache_vs_ttft.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return [path]

    # ── Figure 5: TPS by output length ────────────────────────────────────────
    def fig_tps_by_output_length(self) -> list[str]:
        """Line plot: tokens/sec vs target output length per strategy."""
        df = self._df[self._df["tokens_per_second"] > 0].copy()
        strategies = df["strategy_name"].unique()
        palette = sns.color_palette(PALETTE, len(strategies))

        fig, ax = plt.subplots(figsize=(7, 4))
        for i, strat in enumerate(sorted(strategies)):
            sdf = (
                df[df["strategy_name"] == strat]
                .groupby("target_output_tokens")["tokens_per_second"]
                .median()
                .reset_index()
            )
            ax.plot(
                sdf["target_output_tokens"], sdf["tokens_per_second"],
                marker="o", label=strat, color=palette[i], linewidth=1.5, markersize=4,
            )

        ax.set_xlabel("Target Output Length (tokens)")
        ax.set_ylabel("Tokens / Second (median)")
        ax.set_title("Decode Throughput vs. Output Length by Strategy")
        ax.legend(fontsize=7)
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "tps_by_output_length.pdf")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return [path]

    # ── Table 1: Summary LaTeX ────────────────────────────────────────────────
    def table_summary_latex(self) -> list[str]:
        """LaTeX table: TTFT/E2E P50/P95/P99 per strategy."""
        if not self._summaries:
            return []

        rows = []
        for s in self._summaries:
            rows.append({
                "Strategy": s.get("strategy", ""),
                "TTFT P50": f"{s.get('ttft_p50_ms', 0):.0f}",
                "TTFT P95": f"{s.get('ttft_p95_ms', 0):.0f}",
                "E2E P50":  f"{s.get('e2e_p50_ms', 0):.0f}",
                "E2E P95":  f"{s.get('e2e_p95_ms', 0):.0f}",
                "TPS":      f"{s.get('tps_mean', 0):.1f}",
                "N":        str(s.get("n_ok", 0)),
            })

        df = pd.DataFrame(rows)
        latex = df.to_latex(
            index=False, escape=False, column_format="lrrrrrr",
            caption="Benchmark summary: latency (ms) and throughput (tokens/s) per strategy.",
            label="tab:summary",
        )

        path = os.path.join(self.table_dir, "summary.tex")
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Saved {path}")
        return [path]

    # ── Table: efficiency / SLO / mechanism metrics ───────────────────────────
    def table_efficiency_latex(self) -> list[str]:
        """LaTeX table: SLO attainment, queueing delay, ITL, preemptions, energy, completion ratio."""
        if not self._summaries:
            return []

        slo_key = next((k for k in self._summaries[0] if k.startswith("slo_attainment_ttft_")), None)

        rows = []
        for s in self._summaries:
            row = {
                "Strategy": s.get("strategy", "").replace("_", "\\_"),
                "SLO Attain. (\\%)": f"{s.get(slo_key, 0):.1f}" if slo_key else "—",
                "Queue Delay P50 (ms)": f"{s.get('queue_delay_p50_ms', 0):.0f}",
                "ITL Mean (ms)": f"{s.get('itl_mean_ms', 0):.1f}" if "itl_mean_ms" in s else "—",
                "Preemptions": str(s.get("total_preemptions", 0)),
                "Energy/Token (J)": f"{s.get('energy_per_token_j', 0):.3f}" if "energy_per_token_j" in s else "—",
                "Completion Ratio": f"{s.get('completion_ratio_mean', 0):.2f}",
            }
            if "spec_draft_acceptance_rate_mean" in s:
                row["Spec. Accept. Rate"] = f"{s['spec_draft_acceptance_rate_mean']:.2f}"
            rows.append(row)

        df = pd.DataFrame(rows)
        latex = df.to_latex(
            index=False, escape=False,
            caption="Mechanistic and efficiency metrics: SLO attainment (TTFT-based), "
                    "queueing delay, inter-token latency, preemption count, energy per "
                    "output token, and output completion ratio.",
            label="tab:efficiency",
        )

        path = os.path.join(self.table_dir, "efficiency.tex")
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Saved {path}")
        return [path]

    # ── Table 2: Interaction effects ──────────────────────────────────────────
    def table_interaction_effects(self) -> list[str]:
        """LaTeX table: improvement vs baseline for combined strategies."""
        df = self._df
        baseline_p95 = (
            df[df["strategy_name"].str.startswith("baseline")]["ttft_ms"]
            .quantile(0.95)
        )
        if pd.isna(baseline_p95):
            return []

        rows = []
        for strat in sorted(df["strategy_name"].unique()):
            p95 = df[df["strategy_name"] == strat]["ttft_ms"].quantile(0.95)
            delta = baseline_p95 - p95
            rows.append({
                "Strategy": strat.replace("_", "\\_"),
                "TTFT P95 (ms)": f"{p95:.0f}",
                r"$\Delta$ vs Baseline": f"{delta:+.0f}",
                r"Improvement (\%)": f"{delta/baseline_p95*100:+.1f}",
            })

        latex = pd.DataFrame(rows).to_latex(
            index=False, escape=False,
            caption="Strategy interaction effects relative to full-prefill fp16 TP-1 baseline.",
            label="tab:interactions",
        )
        path = os.path.join(self.table_dir, "interactions.tex")
        with open(path, "w") as f:
            f.write(latex)
        logger.info(f"Saved {path}")
        return [path]
