import argparse
import errno
import logging
from pathlib import Path

from webapp.server import run_browser_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Start the Automated Eligibility Criteria Extraction backend API "
            "and browser interface."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--manifest", type=Path,
        help="Run or resume NCT IDs from a JSON manifest instead of starting the browser app.",
    )
    args = parser.parse_args()
    if args.manifest:
        from backend.batch_manifest import run_manifest
        result = run_manifest(args.manifest)
        print(f"Batch state: {result['state_file']}")
        return
    try:
        run_browser_app(args.host, args.port, not args.no_browser)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            parser.exit(
                1,
                f"Cannot start: {args.host}:{args.port} is already in use.\n"
                f"Modern AutoCrit may already be running at "
                f"http://{args.host}:{args.port}.\n",
            )
        raise


if __name__ == "__main__":
    main()
