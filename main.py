import argparse

from webapp.server import run_browser_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Modern AutoCrit in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    run_browser_app(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
