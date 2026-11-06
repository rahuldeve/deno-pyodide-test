"""`python -m pyodide_sandbox fetch`: download the Pyodide files ahead of time."""

import argparse
import logging
import sys

from pyodide_sandbox.distribution import (
    PYODIDE_VERSION,
    DistributionError,
    default_dist_dir,
    default_index_url,
    ensure_distribution,
)
from pyodide_sandbox.session import DEFAULT_PACKAGES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pyodide_sandbox")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch",
        help=f"download the Pyodide {PYODIDE_VERSION} files that sessions need",
        description="Download the Pyodide runtime and the given packages, with "
        "their dependencies, so sessions start without network access.",
    )
    fetch.add_argument(
        "packages",
        nargs="*",
        default=list(DEFAULT_PACKAGES),
        help=f"default: {' '.join(DEFAULT_PACKAGES)}",
    )
    fetch.add_argument(
        "--dir", help=f"where to put them (default: {default_dist_dir()})"
    )
    fetch.add_argument(
        "--index-url", help=f"a mirror of the release (default: {default_index_url()})"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        root = ensure_distribution(args.packages, args.dir, index_url=args.index_url)
    except (DistributionError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in root.iterdir() if f.is_file())
    print(
        f"Pyodide {PYODIDE_VERSION} with {', '.join(args.packages)}: {root} "
        f"({size / 1e6:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
