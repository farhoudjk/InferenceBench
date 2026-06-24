"""
Results storage and loading.

Per-request results → CSV (one row per request, appendable)
Aggregate summaries  → JSON (one record per strategy × workload)
Raw GPU samples      → CSV (one row per GPU sample)
"""

import csv
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from benchmark.runner import BenchmarkResult, RequestResult
from benchmark.gpu_monitor import GPUMonitorResult

logger = logging.getLogger(__name__)

REQUEST_FIELDS = [
    "request_id", "strategy_name", "workload_name", "task_type",
    "prompt_token_len", "target_output_tokens", "arrival_time",
    "send_time", "ttft_s", "e2e_s", "output_tokens", "tokens_per_second",
    "status", "error", "kv_cache_usage_pct", "num_waiting_seqs",
    "num_running_seqs", "num_preemptions", "num_preemptions_delta",
    "spec_draft_acceptance_rate", "spec_decode_efficiency",
    "ttfb_s", "itl_mean_ms", "itl_p50_ms", "itl_p95_ms", "itl_max_ms",
    "completion_ratio",
]

GPU_SAMPLE_FIELDS = [
    "strategy_name", "timestamp", "gpu_index",
    "util_compute_pct", "util_memory_pct",
    "mem_used_mib", "mem_total_mib", "power_w", "temperature_c",
]


class ResultStore:

    def __init__(self, results_dir: str = "results", run_id: Optional[str] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(results_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.requests_csv   = os.path.join(self.run_dir, "requests.csv")
        self.gpu_csv        = os.path.join(self.run_dir, "gpu_samples.csv")
        self.summary_json   = os.path.join(self.run_dir, "summary.json")

        self._requests_header_written = os.path.exists(self.requests_csv)
        self._gpu_header_written      = os.path.exists(self.gpu_csv)
        self._summaries: list[dict]   = self._load_existing_summaries()

        logger.info(f"ResultStore initialized at {self.run_dir}")

    def _load_existing_summaries(self) -> list[dict]:
        if os.path.exists(self.summary_json):
            with open(self.summary_json) as f:
                return json.load(f)
        return []

    def save(self, result: BenchmarkResult) -> None:
        """Persist a completed BenchmarkResult."""
        self._save_requests(result.requests)
        if result.gpu_result:
            self._save_gpu_samples(result.gpu_result)
        self._save_summary(result)
        logger.debug(
            f"Saved {len(result.requests)} request records "
            f"for [{result.strategy.name}]"
        )

    def _save_requests(self, requests: list[RequestResult]) -> None:
        write_header = not self._requests_header_written
        with open(self.requests_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUEST_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._requests_header_written = True
            for req in requests:
                writer.writerow({k: getattr(req, k, "") for k in REQUEST_FIELDS})

    def _save_gpu_samples(self, gpu_result: GPUMonitorResult) -> None:
        write_header = not self._gpu_header_written
        with open(self.gpu_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=GPU_SAMPLE_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._gpu_header_written = True
            for sample in gpu_result.samples:
                row = asdict(sample)
                row["strategy_name"] = gpu_result.strategy_name
                writer.writerow({k: row.get(k, "") for k in GPU_SAMPLE_FIELDS})

    def _save_summary(self, result: BenchmarkResult) -> None:
        summary = result.summary()
        summary["run_id"] = self.run_id
        summary["vllm_startup_time_s"] = result.vllm_startup_time_s
        if result.gpu_result:
            gpu_summary = result.gpu_result.summary()
            summary["gpu"] = gpu_summary
            # Energy per output token: mean power (W) * wall time (s) = Joules,
            # divided by total tokens generated across all successful requests.
            total_tokens = sum(r.output_tokens for r in result.successful())
            if gpu_summary and total_tokens > 0 and result.wall_time_s > 0:
                energy_j = gpu_summary["power_mean_w"] * result.wall_time_s
                summary["energy_per_token_j"] = energy_j / total_tokens

        self._summaries.append(summary)
        with open(self.summary_json, "w") as f:
            json.dump(self._summaries, f, indent=2)

    def load_summaries(self) -> list[dict]:
        return self._summaries

    def load_requests_csv(self) -> str:
        return self.requests_csv

    def run_dir_path(self) -> str:
        return self.run_dir
