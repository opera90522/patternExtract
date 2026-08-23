"""Command line interface: ``patgen learn | match | inspect | bench``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence

from .io_csv import read_texts
from .learn import LearnConfig, TemplateLearner
from .matcher import TemplateMatcher
from .report import coverage_report, render_library
from .template import TemplateLibrary


def _learn(args: argparse.Namespace) -> int:
    texts = list(read_texts(args.input, args.column, args.limit))
    if not texts:
        print("no rows read", file=sys.stderr)
        return 1
    config = LearnConfig(
        sim_threshold=args.sim,
        depth=args.depth,
        min_support=args.min_support,
    )
    started = time.perf_counter()
    library = TemplateLearner(config).fit(texts)
    elapsed = time.perf_counter() - started
    library.meta["learn_seconds"] = round(elapsed, 3)
    library.save(args.out)
    stats = coverage_report(library, texts)
    library.meta.update(stats)
    library.save(args.out)
    print(
        f"{len(texts)} messages -> {len(library)} templates in {elapsed:.2f}s "
        f"(coverage {stats['coverage']:.1%}) -> {args.out}"
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(render_library(library, stats))
        print(f"report -> {args.report}")
    return 0


def _match(args: argparse.Namespace) -> int:
    library = TemplateLibrary.load(args.model)
    matcher = TemplateMatcher(library)
    texts = read_texts(args.input, args.column, args.limit)
    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    matched = total = 0
    try:
        for text in texts:
            total += 1
            result = matcher.match(text)
            matched += result is not None
            record = {"raw": text, **(result.as_dict() if result else {"template_id": None})}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if args.out:
            out.close()
    print(f"matched {matched}/{total} ({matched / max(total, 1):.1%})", file=sys.stderr)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    library = TemplateLibrary.load(args.model)
    print(render_library(library, library.meta))
    return 0


def _bench(args: argparse.Namespace) -> int:
    library = TemplateLibrary.load(args.model)
    matcher = TemplateMatcher(library)
    texts = list(read_texts(args.input, args.column, args.limit))
    if not texts:
        print("no rows read", file=sys.stderr)
        return 1
    texts = (texts * (args.repeat or 1))
    started = time.perf_counter()
    matched = sum(matcher.match(t) is not None for t in texts)
    elapsed = time.perf_counter() - started
    print(
        f"{len(texts)} messages in {elapsed:.3f}s = {len(texts) / elapsed:,.0f} msg/s "
        f"({matched / len(texts):.1%} matched, {len(library)} templates)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patgen", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn", help="mine templates from CSV files")
    learn.add_argument("input", nargs="+", help="CSV paths or globs")
    learn.add_argument("-c", "--column", help="text column (auto-detected if omitted)")
    learn.add_argument("-o", "--out", default="templates.json")
    learn.add_argument("--sim", type=float, default=0.5, help="similarity threshold")
    learn.add_argument("--depth", type=int, default=4)
    learn.add_argument("--min-support", type=int, default=1)
    learn.add_argument("--limit", type=int)
    learn.add_argument("--report", help="write a markdown report here")
    learn.set_defaults(func=_learn)

    match = sub.add_parser("match", help="match messages against a template library")
    match.add_argument("input", nargs="+")
    match.add_argument("-m", "--model", required=True)
    match.add_argument("-c", "--column")
    match.add_argument("-o", "--out", help="JSONL output (default stdout)")
    match.add_argument("--limit", type=int)
    match.set_defaults(func=_match)

    inspect = sub.add_parser("inspect", help="print a learned library")
    inspect.add_argument("-m", "--model", required=True)
    inspect.set_defaults(func=_inspect)

    bench = sub.add_parser("bench", help="measure production throughput")
    bench.add_argument("input", nargs="+")
    bench.add_argument("-m", "--model", required=True)
    bench.add_argument("-c", "--column")
    bench.add_argument("--limit", type=int)
    bench.add_argument("--repeat", type=int, default=1)
    bench.set_defaults(func=_bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
