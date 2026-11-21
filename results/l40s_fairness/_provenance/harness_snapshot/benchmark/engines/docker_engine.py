"""
Shared support for Docker-container-based engines (currently TGI).

A container-based engine's server process isn't a child of this Python
process the way vLLM/SGLang/TensorRT-LLM's subprocess.Popen launches are —
`docker run -d` starts a detached container and returns immediately, and
tearing it down means `docker stop`/`docker rm`, not signaling a PID. This
module provides a DockerContainerHandle satisfying the same ProcessHandle
protocol base.py's shared start()/stop()/_wait_for_health() already use, so
those methods work unmodified for a container-based Engine subclass.

*** WARNING: `docker run --gpus all` has hung this host twice ***
See tgi_engine.py's module docstring for the full incident detail (2026-07-22:
TGI hung at model warmup with --gpus all, took the whole host down hard
enough to drop the operator's SSH session, both with and without CUDA graph
capture). Not confirmed whether the risk is specific to TGI's image or a
more general property of `--gpus all` container GPU passthrough on this
host's driver/nvidia-container-toolkit combination. Any new engine built on
run_container() here should be smoke-tested with an external kill-switch
(force-kill the container after N seconds of zero GPU utilization) rather
than trusting base.py's health-poll timeout alone — that timeout only
protects the polling asyncio task, not the host, if the container itself
wedges the GPU driver.
"""

import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

DOCKER = "docker"   # assumes passwordless docker (user in the docker group)
                     # or sudo configured for it; see this host's setup notes.


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


class DockerContainerHandle:
    """ProcessHandle wrapping a detached (`docker run -d`) container."""

    def __init__(self, container_name: str):
        self.container_name = container_name
        self._returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        """Returns None if the container is running, else its exit code
        (0 if it can't be determined, e.g. already removed)."""
        result = _run([DOCKER, "inspect", "--format",
                       "{{.State.Running}} {{.State.ExitCode}}", self.container_name])
        if result.returncode != 0:
            # Container doesn't exist (removed or never created) — treat as exited.
            self._returncode = self._returncode if self._returncode is not None else 0
            return self._returncode
        parts = result.stdout.strip().split()
        if len(parts) != 2:
            return None
        running, exit_code = parts
        if running == "true":
            return None
        self._returncode = int(exit_code)
        return self._returncode

    @property
    def returncode(self) -> Optional[int]:
        return self._returncode

    def terminate(self) -> None:
        _run([DOCKER, "stop", "--time", "15", self.container_name])

    def kill(self) -> None:
        _run([DOCKER, "kill", self.container_name])
        _run([DOCKER, "rm", "-f", self.container_name])

    def wait(self, timeout: Optional[float] = None) -> int:
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.time() >= deadline:
                raise subprocess.TimeoutExpired(cmd=f"docker wait {self.container_name}", timeout=timeout)
            time.sleep(1)

    def logs(self) -> str:
        result = _run([DOCKER, "logs", self.container_name])
        return result.stdout + result.stderr


def run_container(
    image: str,
    container_name: str,
    port_map: dict[int, int],
    volumes: dict[str, str],
    env: dict[str, str],
    cmd_args: list[str],
    gpus: str = "all",
) -> DockerContainerHandle:
    """Launch a detached GPU-enabled container. port_map is {host: container},
    volumes is {host_path: container_path}."""
    # Best-effort cleanup of a same-named container left over from a crashed
    # prior run — `docker run` fails outright if the name is already taken.
    _run([DOCKER, "rm", "-f", container_name])

    cmd = [DOCKER, "run", "-d", "--gpus", gpus, "--name", container_name]
    for host_port, container_port in port_map.items():
        cmd += ["-p", f"{host_port}:{container_port}"]
    for host_path, container_path in volumes.items():
        cmd += ["-v", f"{host_path}:{container_path}"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)
    cmd += cmd_args

    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"docker run failed for container {container_name!r}: {result.stderr.strip()}"
        )
    return DockerContainerHandle(container_name)


def stream_logs_to_file(container_name: str, log_path: str) -> subprocess.Popen:
    """Start a background `docker logs -f` piping the container's output into
    log_path, mirroring how subprocess-based engines write their own log file
    directly. Caller is responsible for terminating this tail process."""
    log_file = open(log_path, "w")
    return subprocess.Popen(
        [DOCKER, "logs", "-f", container_name],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
