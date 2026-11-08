"""Thin CLI around scripts.jev.pipeline.run_model.

Usage:
    python -m scripts.jev.run                              # judge all 9 models
    python -m scripts.jev.run --model claude-sonnet-5      # single model
    python -m scripts.jev.run --model claude-sonnet-5 --limit 5 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio

from scripts.jev.pipeline import HUMAN_EVAL_FILES, run_model


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run JEV (noul mode) over the human-eval set.")
    p.add_argument("--model", choices=sorted(HUMAN_EVAL_FILES),
                   help="Model key to judge. Omit to run all nine.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only judge the first N rows (smoke test).")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Parallel JEV requests in flight.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


async def _main() -> None:
    args = _parse()
    targets = [args.model] if args.model else list(HUMAN_EVAL_FILES)
    for key in targets:
        await run_model(
            key,
            concurrency=args.concurrency,
            limit=args.limit,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    asyncio.run(_main())
