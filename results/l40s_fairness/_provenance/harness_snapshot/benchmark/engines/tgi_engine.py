"""
TGI (Hugging Face Text Generation Inference) engine adapter.

Unlike vLLM/SGLang/TensorRT-LLM, TGI has no supported pip-installable
server package on this host (no prebuilt wheel; building from source needs
a Rust toolchain) — it ships as a Docker image
(ghcr.io/huggingface/text-generation-inference). This adapter launches it
via `docker run -d --gpus all` instead of a subprocess.Popen, using
docker_engine.py's DockerContainerHandle so the shared start()/stop()/
_wait_for_health() lifecycle in base.py works unmodified.

Flags verified against a live `docker run ... --help` (image tag
`latest`, router v3.3.6-dev0) and a live /metrics scrape from an actual
served request (TheBloke/Llama-2-7B-Chat-AWQ, --quantize awq) on this host.

Two real surprises from that live verification, both load-bearing for this
adapter's design:
  1. --prometheus-port is accepted by the CLI but does not actually bind —
     /metrics is served on the same port as the main API (--port), not the
     documented separate port. Confirmed via `docker exec ... cat
     /proc/net/tcp`: only the main port's socket exists.
  2. --dtype and --quantize are mutually exclusive per TGI's own --help
     text ("This option cannot be used with --quantize") — build_args()
     sends at most one of them, never both.

*** WARNING — UNSAFE ON THIS HOST, DO NOT RUN WITHOUT INVESTIGATING FIRST ***
Verified twice (2026-07-22, TheBloke/Llama-2-7B-Chat-AWQ on an NVIDIA L40S,
driver 570.211.01, nvidia-container-toolkit 1.19.1): the container reliably
hangs at router startup's "Warming up model" step (GPU utilization drops to
0% and never recovers) and BOTH TIMES took the entire host down hard enough
to drop the operator's SSH session — not just the container. Retried with
--cuda-graphs 0 (disabling CUDA graph capture, the leading hypothesis for a
container-GPU-passthrough interaction bug) and got the identical hang, which
rules out CUDA graph capture as the sole cause. Root cause not identified —
could be this specific TGI image build, this driver/nvidia-container-toolkit
version combination, or something else about this host's GPU passthrough
config. vLLM, SGLang, and TensorRT-LLM all run fine natively on this same
GPU, so the failure is specific to TGI's Docker + --gpus all path, not the
hardware in general.

Do not re-run this adapter on this host without: (a) testing on different
hardware/driver/toolkit versions first to isolate the variable, or (b)
adding a hard automated kill-switch (e.g. `docker run --stop-timeout` plus
an external watchdog that force-kills the container after N seconds of
zero GPU utilization post-launch) so a hang can't take the host down again.
"""

import logging
import os
from typing import Optional

import aiohttp

from benchmark.engines.base import Engine, EngineInstance, ServerMetrics
from benchmark.engines.docker_engine import DockerContainerHandle, run_container, stream_logs_to_file
from config.strategies import Strategy

logger = logging.getLogger(__name__)

TGI_IMAGE = os.environ.get("TGI_IMAGE", "ghcr.io/huggingface/text-generation-inference:latest")

# TGI only accepts float16/bfloat16 for --dtype (no "auto"); map Strategy's
# default "float16" straight through, and skip the flag entirely for
# anything TGI doesn't recognize rather than pass through a value it'll reject.
_TGI_DTYPES = {"float16", "bfloat16"}


class TGIEngine(Engine):
    name = "tgi"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_tail: Optional[object] = None  # subprocess.Popen from stream_logs_to_file

    def preflight_check(self, strategy: Strategy) -> None:
        if strategy.tensor_parallel_size > 1:
            # TGI shards via --num-shard / SHARDED env, not a --tensor-parallel-size
            # flag — not yet mapped in build_args(). Fail fast with a clear reason
            # rather than silently ignoring the strategy's TP setting.
            raise RuntimeError(
                f"TGIEngine does not yet support tensor_parallel_size>1 "
                f"(strategy={strategy.name!r} requests TP={strategy.tensor_parallel_size}); "
                "TGI's sharding flag (--num-shard) isn't wired into build_args() yet."
            )

    def _container_name(self, strategy: Strategy, port: int) -> str:
        return f"inferencebench-tgi-{strategy.name}-{port}"

    def build_args(self, strategy: Strategy, port: int) -> list[str]:
        args = [
            "--model-id", self.model,
            "--port", "80",   # internal container port; mapped to `port` on the host
            "--max-total-tokens", str(strategy.max_model_len),
        ]

        # --dtype and --quantize are mutually exclusive per TGI's CLI (see
        # module docstring) — but this is stricter than it first looks:
        # verified live that TGI auto-detects quantization from the
        # checkpoint's own config (e.g. an AWQ-quantized HF repo) regardless
        # of whether --quantize was passed, and then rejects an explicit
        # --dtype as conflicting with that auto-detected value. So skip
        # --dtype whenever the model id itself signals a quantized
        # checkpoint, not only when strategy.quantization is set.
        model_lower = self.model.lower()
        is_quantized_checkpoint = (
            bool(strategy.quantization)
            or any(k in model_lower for k in ("awq", "gptq", "-fp8", "marlin"))
        )
        if strategy.quantization:
            args += ["--quantize", strategy.quantization]
        elif not is_quantized_checkpoint and strategy.dtype in _TGI_DTYPES:
            args += ["--dtype", strategy.dtype]

        if strategy.max_num_batched_tokens is not None:
            args += ["--max-batch-prefill-tokens", str(strategy.max_num_batched_tokens)]
        if strategy.max_num_seqs is not None:
            args += ["--max-concurrent-requests", str(strategy.max_num_seqs)]

        if strategy.enable_speculation and strategy.ngram_prompt_lookup_max:
            # TGI's --speculate is n-gram speculation with a single "number of
            # tokens to speculate" knob — no separate prompt-lookup-max control
            # like vLLM/SGLang. Map num_speculative_tokens; there's no
            # equivalent for ngram_prompt_lookup_max to warn about losing,
            # since TGI's ngram implementation doesn't expose that dimension
            # at all (same "each engine's own mechanism" caveat as the
            # SGLang/TensorRT-LLM ngram adapters).
            args += ["--speculate", str(strategy.num_speculative_tokens)]
        elif strategy.enable_speculation and strategy.speculative_model:
            logger.warning(
                f"[{strategy.name}] TGI's --speculate is n-gram-only in this adapter; "
                f"draft-model speculative decoding (speculative_model={strategy.speculative_model!r}) "
                "has no mapping here and is left unconfigured."
            )

        return args

    def _launch_process(self, strategy: Strategy, port: int) -> DockerContainerHandle:
        container_name = self._container_name(strategy, port)
        log_path = self._log_path(strategy)

        logger.info(f"Launching tgi [{strategy.name}] on port {port} (container {container_name})")
        logger.info(f"Image:   {TGI_IMAGE}")
        args = self.build_args(strategy, port)
        logger.info(f"Args:    {' '.join(args)}")
        logger.info(f"Log:     {log_path}")

        handle = run_container(
            image=TGI_IMAGE,
            container_name=container_name,
            port_map={port: 80},
            # Mount the host's HF cache read/write so TGI reuses already-
            # downloaded checkpoints instead of re-fetching inside the container.
            volumes={os.path.expanduser("~/.cache/huggingface"): "/data"},
            env={},
            cmd_args=args,
        )
        self._log_tail = stream_logs_to_file(container_name, log_path)
        return handle

    def stop(self, instance: Optional[EngineInstance] = None) -> None:
        super().stop(instance)
        if self._log_tail is not None:
            self._log_tail.terminate()
            self._log_tail = None
        # Belt-and-suspenders: base.stop() calls handle.terminate()/kill(),
        # which map to `docker stop`/`docker kill`, but neither removes the
        # stopped container — do that here so a re-run with the same
        # strategy/port doesn't collide on the container name.
        target = instance or self._current_instance
        if target is not None and isinstance(target.process, DockerContainerHandle):
            from benchmark.engines.docker_engine import _run, DOCKER
            _run([DOCKER, "rm", "-f", target.process.container_name])

    @classmethod
    def process_match_patterns(cls) -> list[str]:
        # TGI's frontend isn't a Python subprocess on the host at all (it's
        # inside a container) — kill_stale_gpu_workers()'s pgrep-based sweep
        # can't see it. Stale TGI containers are cleaned up via `docker rm -f`
        # in stop() instead; nothing to add here.
        return []

    async def get_server_metrics(self, instance: EngineInstance) -> ServerMetrics:
        """Fetch Prometheus metrics from TGI's /metrics endpoint.

        Verified live: despite --prometheus-port being accepted, /metrics is
        actually served on the main API port (see module docstring). Only
        tgi_queue_size / tgi_batch_current_size were confirmed present in a
        live scrape; TGI exposes no KV-cache-percentage gauge, no preemption
        counter, and no speculative-decode acceptance-rate metric.
        """
        target_metrics = {"tgi_queue_size", "tgi_batch_current_size"}

        raw: dict = {}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    instance.metrics_url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return ServerMetrics()
                    text = await resp.text()

            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                for key in target_metrics:
                    if key in raw:
                        continue
                    if line.startswith(key + " ") or line.startswith(key + "{"):
                        parts = line.rsplit(" ", 1)
                        if len(parts) == 2:
                            try:
                                raw[key] = float(parts[1])
                            except ValueError:
                                pass
        except Exception:
            return ServerMetrics()

        return ServerMetrics(
            kv_cache_usage_pct=None,   # no equivalent gauge exposed by TGI
            num_waiting_seqs=_as_int(raw.get("tgi_queue_size")),
            num_running_seqs=_as_int(raw.get("tgi_batch_current_size")),
            num_preemptions_total=None,
            spec_draft_acceptance_rate=None,
            spec_decode_efficiency=None,
            raw=raw,
        )


def _as_int(value: Optional[float]) -> Optional[int]:
    return int(value) if value is not None else None
