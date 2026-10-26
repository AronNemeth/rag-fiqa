"""Unified launcher that starts both the FastAPI backend and Streamlit frontend."""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

import httpx

from rag_fiqa.config import API_HOST, API_PORT, UI_HOST, UI_PORT
from rag_fiqa.utils import get_logger, setup_logging

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command line options for the unified launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default=API_HOST, help=f"Backend API host (default: {API_HOST}).")
    parser.add_argument("--api-port", type=int, default=API_PORT, help=f"Backend API port (default: {API_PORT}).")
    parser.add_argument("--ui-host", default=UI_HOST, help=f"Frontend UI host (default: {UI_HOST}).")
    parser.add_argument("--ui-port", type=int, default=UI_PORT, help=f"Frontend UI port (default: {UI_PORT}).")
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the backend /health endpoint before timing out (default: 30.0).",
    )
    return parser.parse_args()


def wait_for_backend(url: str, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
    """Poll the backend health endpoint until it responds with HTTP 200.

    Returns:
        True if the backend is healthy within the timeout, False otherwise.
    """
    deadline = time.perf_counter() + timeout
    health_url = f"{url.rstrip('/')}/health"

    while time.perf_counter() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(health_url)
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("status") == "ok":
                        return True
        except httpx.ConnectError, httpx.TimeoutException, OSError:
            pass
        time.sleep(poll_interval)
    return False


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Gracefully terminate a list of running child processes."""
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    for proc in processes:
        try:
            proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def run() -> None:
    """Start backend and frontend child processes and manage their lifecycles."""
    args = parse_args()
    setup_logging()

    api_url = f"http://127.0.0.1:{args.api_port}"
    ui_url = f"http://localhost:{args.ui_port}"

    print("\n" + "=" * 65)
    print("  Starting FiQA Financial Q&A Application")
    print(f"  - Backend API : {api_url}")
    print(f"  - Frontend UI : {ui_url}")
    print("=" * 65 + "\n")

    api_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "serve_api.py"),
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]

    ui_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "serve_ui.py"),
        "--host",
        args.ui_host,
        "--port",
        str(args.ui_port),
    ]

    active_processes: list[subprocess.Popen[bytes]] = []

    def handle_exit(signum: int, frame: FrameType | None) -> None:
        print("\nStopping application services...")
        terminate_processes(active_processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # 1. Start FastAPI backend
    logger.info("Starting FastAPI backend server...")
    api_proc = subprocess.Popen(api_cmd)
    active_processes.append(api_proc)

    # 2. Wait for backend to be healthy
    logger.info("Waiting for backend vector store and models to initialize...")
    is_ready = wait_for_backend(api_url, timeout=args.health_timeout)
    if not is_ready:
        if api_proc.poll() is not None:
            logger.error("FastAPI process exited prematurely with code %d", api_proc.returncode)
        else:
            logger.error("Backend health check timed out after %.1fs", args.health_timeout)
        terminate_processes(active_processes)
        sys.exit(1)

    print("Backend API is ready.")

    # 3. Start Streamlit UI
    logger.info("Starting Streamlit frontend...")
    ui_proc = subprocess.Popen(ui_cmd)
    active_processes.append(ui_proc)

    print(f"\n Application is running! Open your browser at: {ui_url}\n")
    print("Press Ctrl+C to stop all services.\n")

    # 4. Monitor both processes until exit
    try:
        while True:
            if api_proc.poll() is not None:
                logger.error("Backend process exited unexpectedly (code: %s)", api_proc.returncode)
                break
            if ui_proc.poll() is not None:
                logger.info("Streamlit process exited (code: %s)", ui_proc.returncode)
                break
            time.sleep(1.0)
    finally:
        terminate_processes(active_processes)


if __name__ == "__main__":
    run()
