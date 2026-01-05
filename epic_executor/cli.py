#!/usr/bin/env python3
"""CLI for epic executor."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Execute tasks from an epic folder in parallel"
    )
    parser.add_argument(
        "epic_folder",
        help="Path to folder containing numbered task .md files",
    )
    parser.add_argument(
        "--project-root",
        "-p",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=4,
        help="Maximum concurrent agents (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Parse and show plan without executing",
    )

    args = parser.parse_args()

    epic_folder = Path(args.epic_folder).resolve()
    project_root = Path(args.project_root).resolve()

    if not epic_folder.exists():
        print(f"Error: Epic folder not found: {epic_folder}")
        sys.exit(1)

    if args.dry_run:
        from .executor import dry_run
        asyncio.run(dry_run(str(epic_folder)))
    else:
        from .executor import execute_epic
        status = asyncio.run(
            execute_epic(
                str(epic_folder),
                str(project_root),
                max_concurrent=args.max_concurrent,
            )
        )
        sys.exit(0 if not status.failed else 1)


if __name__ == "__main__":
    main()
