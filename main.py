"""
LLM Execution Strategy Benchmark — Main Orchestrator

Usage examples:
  # Run full benchmark (all strategies × all workloads)
  python main.py --model meta-llama/Llama-3.1-8B-Instruct --run-all

  # Run single strategy × workload (for debugging)
  python main.py --model meta-llama/Llama-3.1-8B-Instruct \
                 --strategy baseline_full_fp16_tp1 \
                 --workload chat_poisson_low

  # Run dimension ablation (vary one dimension at a time)
  python main.py --model meta-llama/Llama-3.1-8B-Instruct \
                 --dimension chunked_prefill \
                 --workload chat_poisson_high

  # Generate report from existing results
  python main.py --report-only --run-id 20260526_120000
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
from datetime import datetime

from benchmark.analysis.cost_model import CostModelFitter
from benchmark.analysis.report import ReportGenerator
from benchmark.gpu_monitor import GPUMonitor
from benchmark.launcher import VLLMLauncher, kill_stale_gpu_workers
from benchmark.runner import run_benchmark
from benchmark.storage import ResultStore
from config.strategies import STRATEGIES, get_strategies_for_dimension
from config.workloads import WORKLOADS, get_workload

def _setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    if not debug:
        # suppress noisy debug from aiohttp internals
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
logger = logging.getLogger("benchmark.main")


def parse_args():
    p = argparse.ArgumentParser(description="LLM Execution Strategy Benchmark")
    p.add_argument("--model",       required=True, help="HuggingFace model ID or local path")
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--run-id",      default=None,    help="Run ID (default: timestamp)")
    p.add_argument("--gpu-indices", default="0",     help="Comma-separated GPU indices")
    p.add_argument("--port",        default=8100, type=int)
    p.add_argument("--max-concurrent", default=16, type=int, help="Max concurrent requests")
    p.add_argument("--request-timeout", default=300, type=float)
    p.add_argument("--seed",        default=42, type=int)

    # What to run
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-all", action="store_true", help="All strategies × all workloads")
    g.add_argument("--strategy", help="Single strategy name")
    g.add_argument("--dimension", choices=["chunked_prefill","tensor_parallel","speculation","quantization","interaction"])
    g.add_argument("--report-only", action="store_true")

    p.add_argument("--workload", default="chat_poisson_low", help="Workload name")
    p.add_argument("--workloads", nargs="+", help="Multiple workloads")
    p.add_argument("--no-gpu-monitor", action="store_true")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging for request-level diagnostics")
    p.add_argument("--num-requests", type=int, default=None,
                   help="Override num_requests for all selected workloads")
    p.add_argument("--extra-vllm-args", nargs="+", default=[])

    return p.parse_args()


async def run(args):
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    store  = ResultStore(results_dir=args.results_dir, run_id=run_id)
    gpu_indices = [int(x) for x in args.gpu_indices.split(",")]

    # Environment diagnostics
    import torch
    logger.info(f"Python:      {sys.executable}")
    logger.info(f"PyTorch:     {torch.__version__}")
    logger.info(f"CUDA built:  {torch.version.cuda}")
    logger.info(f"CUDA avail:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in gpu_indices:
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024 ** 3)
            logger.info(f"GPU {i}:       {props.name} ({mem_gb:.1f} GiB)")
    else:
        logger.warning("CUDA not available — vLLM will fail to start!")

    # Select strategies
    if args.run_all:
        strategies = list(STRATEGIES.values())
        workload_names = args.workloads or list(WORKLOADS.keys())
    elif args.dimension:
        strategies = list(get_strategies_for_dimension(args.dimension).values())
        workload_names = args.workloads or [args.workload]
    elif args.strategy:
        from config.strategies import get_strategy
        strategies = [get_strategy(args.strategy)]
        workload_names = args.workloads or [args.workload]
    else:
        logger.error("No execution mode specified.")
        sys.exit(1)

    workloads = [get_workload(w) for w in workload_names]
    if args.num_requests is not None:
        for wl in workloads:
            logger.info(f"Overriding num_requests for [{wl.name}]: {wl.num_requests} → {args.num_requests}")
            wl.num_requests = args.num_requests
    launcher  = VLLMLauncher(
        model=args.model,
        results_dir=store.run_dir_path(),
        extra_vllm_args=args.extra_vllm_args,
    )

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Strategies: {[s.name for s in strategies]}")
    logger.info(f"Workloads:  {workload_names}")
    logger.info(f"Results:    {store.run_dir_path()}")

    total = len(strategies) * len(workloads)
    done  = 0

    for strategy in strategies:
        for workload in workloads:
            done += 1
            logger.info(
                f"\n{'='*60}\n"
                f"[{done}/{total}] Strategy: {strategy.name}  Workload: {workload.name}\n"
                f"{'='*60}"
            )

            # Skip if tensor parallel requires more GPUs than available
            if strategy.tensor_parallel_size > len(gpu_indices):
                logger.warning(
                    f"Skipping {strategy.name}: requires TP={strategy.tensor_parallel_size} "
                    f"but only {len(gpu_indices)} GPU(s) available."
                )
                continue

            gpu_monitor = (
                GPUMonitor(gpu_indices=gpu_indices)
                if not args.no_gpu_monitor else None
            )

            try:
                instance = await launcher.start(strategy, port=args.port)
                result = await run_benchmark(
                    instance=instance,
                    workload_config=workload,
                    gpu_monitor=gpu_monitor,
                    request_timeout_s=args.request_timeout,
                    max_concurrent=args.max_concurrent,
                    seed=args.seed,
                )
                store.save(result)

            except Exception as e:
                logger.error(f"Failed [{strategy.name}×{workload.name}]: {e}", exc_info=True)
            finally:
                launcher.stop()

    # ── Post-benchmark: analysis and report ──────────────────────────────────
    logger.info("\nGenerating report...")
    reporter = ReportGenerator(run_dir=store.run_dir_path())
    artifacts = reporter.generate_all()

    req_csv = store.load_requests_csv()
    if os.path.exists(req_csv):
        fitter = CostModelFitter(results_dir=store.run_dir_path())
        import pandas as pd
        df = pd.read_csv(req_csv)
        df = df[df["status"] == "ok"]
        if not df.empty:
            analysis_dir = os.path.join(store.run_dir_path(), "analysis")
            fitter.save_analysis(df, output_dir=analysis_dir)
            cost_model = fitter.fit(req_csv)
            logger.info(
                f"\nCost model fitted for {len(cost_model.models)} strategies."
            )

    logger.info(f"\nDone. Results: {store.run_dir_path()}")
    logger.info(f"Artifacts: {len(artifacts)} files generated.")
    print(f"\n✓ Results saved to: {store.run_dir_path()}")


async def report_only(args):
    if not args.run_id:
        logger.error("--report-only requires --run-id")
        sys.exit(1)
    run_dir  = os.path.join(args.results_dir, args.run_id)
    reporter = ReportGenerator(run_dir=run_dir)
    reporter.generate_all()

    req_csv = os.path.join(run_dir, "requests.csv")
    if os.path.exists(req_csv):
        fitter = CostModelFitter(results_dir=run_dir)
        import pandas as pd
        df = pd.read_csv(req_csv)
        df = df[df["status"] == "ok"]
        if not df.empty:
            analysis_dir = os.path.join(run_dir, "analysis")
            fitter.save_analysis(df, output_dir=analysis_dir)

    print(f"✓ Report generated in: {run_dir}")


def _cleanup_gpu_on_exit(signum=None, frame=None):
    """Kill any stale vLLM GPU workers left over from this or a prior crashed run.

    Registered for normal exit (atexit) and SIGINT/SIGTERM (Ctrl+C, kill) so a
    benchmark interrupted mid-run never leaves the GPU occupied for the next one.
    """
    kill_stale_gpu_workers()
    if signum is not None:
        sys.exit(128 + signum)


if __name__ == "__main__":
    args = parse_args()
    _setup_logging(debug=getattr(args, "debug", False))

    # Clear out any orphaned GPU workers from a previous crashed/killed run
    # before we even try to start, then guarantee cleanup on every exit path.
    kill_stale_gpu_workers()
    atexit.register(_cleanup_gpu_on_exit)
    signal.signal(signal.SIGINT, _cleanup_gpu_on_exit)
    signal.signal(signal.SIGTERM, _cleanup_gpu_on_exit)

    if args.report_only:
        asyncio.run(report_only(args))
    else:
        asyncio.run(run(args))
