"""Launch the Streamlit demonstration user interface."""

import argparse
import sys
from pathlib import Path

from streamlit.web import cli as stcli

from rag_fiqa.config import UI_HOST, UI_PORT
from rag_fiqa.utils import setup_logging

UI_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "rag_fiqa" / "ui.py"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Streamlit UI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=UI_HOST, help=f"Host address to bind (default: {UI_HOST}).")
    parser.add_argument("--port", type=int, default=UI_PORT, help=f"Port to listen on (default: {UI_PORT}).")
    return parser.parse_args()


def main() -> None:
    """Configure logging and start Streamlit."""
    args = parse_args()
    setup_logging()
    sys.argv = [
        "streamlit",
        "run",
        str(UI_SCRIPT_PATH),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
