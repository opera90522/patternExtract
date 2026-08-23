"""Generate notebooks/full_pipeline.ipynb with a single end-to-end runnable notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# patgen — full offline pipeline notebook

This notebook runs the complete patgen pipeline on any CSV file:

1. load raw messages and auto-detect the text column
2. normalize, tokenize, and mask entities
3. learn templates with the Drain-style clusterer
4. benchmark production matching throughput
5. profile learning and matching with `cProfile`
6. inspect coverage, top templates, and example matches

**To use your own data** change `CSV_PATH` in the cell below. The container image
in `.devcontainer/` already has Python, Jupyter, and `patgen` installed, so no
network is needed while the notebook runs.""",
    ),
    (
        "code",
        r"""# --- CONFIGURATION: edit these lines for your CSV ------------------------
CSV_PATH = "examples/sms_sample.csv"  # relative to this notebook's directory
TEXT_COLUMN = None                    # e.g. "message"; None = auto-detect
LEXICON = None                        # e.g. "finance" or a path/to/lexicon.json

# Learning knobs
SIM = 0.5
DEPTH = 4
MAX_CHILDREN = 128
MIN_SUPPORT = 1
DROP_DEGENERATE = True
MAX_TEXT_SLOT = 6
REFINE_RATIO = 0.5
LIMIT = None

# Outputs
MODEL_OUT = "model.json"
""",
    ),
    (
        "code",
        r"""import cProfile
import pstats
import random
import time
from collections import Counter
from pathlib import Path

import patgen
from patgen import LearnConfig, TemplateMatcher, learn_templates, list_lexicons
from patgen.io_csv import read_texts
from patgen.report import coverage_report, render_library
""",
    ),
    (
        "code",
        r"""csv_path = Path(CSV_PATH)
if not csv_path.is_absolute():
    csv_path = Path.cwd() / csv_path
    # Handle running from inside the notebooks/ directory
    if not csv_path.exists() and Path.cwd().name == "notebooks":
        csv_path = Path.cwd().parent / CSV_PATH

print("CSV path:", csv_path.resolve())
print("Working directory:", Path.cwd().resolve())
print("patgen version:", patgen.__version__)
print("Available bundled lexicons:", list_lexicons())
""",
    ),
    (
        "code",
        r"""messages = list(read_texts([str(csv_path)], column=TEXT_COLUMN, limit=LIMIT))
print(f"Loaded {len(messages):,} non-empty messages")
print("\n--- first 5 messages ---")
for m in messages[:5]:
    print(m)
""",
    ),
    (
        "md",
        """## 1. Learn templates

`LearnConfig` controls the Drain tree and the optional lexicon.""",
    ),
    (
        "code",
        r"""config = LearnConfig(
    sim_threshold=SIM,
    depth=DEPTH,
    max_children=MAX_CHILDREN,
    min_support=MIN_SUPPORT,
    drop_degenerate=DROP_DEGENERATE,
    max_text_slot_tokens=MAX_TEXT_SLOT,
    refine_ratio=REFINE_RATIO,
    lexicon=LEXICON,
)

t0 = time.perf_counter()
library = learn_templates(messages, config=config)
learn_seconds = time.perf_counter() - t0

print(f"Learned {len(library.templates)} templates from {len(messages):,} messages in {learn_seconds:.2f}s")
if MODEL_OUT:
    out_path = Path(MODEL_OUT)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    library.save(str(out_path))
    print("Saved model to:", out_path.resolve())
""",
    ),
    (
        "md",
        """## 2. Coverage report""",
    ),
    (
        "code",
        r"""cov = coverage_report(library, messages)
print(f"Matched: {cov['matched']:,} / {cov['messages_scored']:,}")
print(f"Coverage: {cov['coverage']:.1%}")
print("\nTop slots by frequency:")
for slot, n in Counter(cov['slot_frequency']).most_common(15):
    print(f"  {slot}: {n:,}")
print("\nUnmatched sample count:", len(cov['unmatched_samples']))
for m in cov['unmatched_samples'][:5]:
    print("  -", m)
""",
    ),
    (
        "md",
        """## 3. Top learned templates""",
    ),
    (
        "code",
        r"""print("Top templates by message count:\n")
for t in library.templates[:15]:
    print(f"{t.template_id} ({t.count:,} msgs): {t.text}")
""",
    ),
    (
        "md",
        """## 4. Benchmark production matching""",
    ),
    (
        "code",
        r"""matcher = TemplateMatcher(library)

# Warm up the matcher once (JIT compilation of regexes happens on first use)
_ = [matcher.match(m) for m in messages[:10]]

# Benchmark: repeat the corpus enough times to get a stable number
repeats = max(3, 10000 // len(messages))
batch = messages * repeats
t0 = time.perf_counter()
for m in batch:
    matcher.match(m)
bench_seconds = time.perf_counter() - t0

print(f"Messages: {len(batch):,}")
print(f"Time: {bench_seconds:.3f}s")
print(f"Throughput: {len(batch) / bench_seconds:,.0f} msg/s")
""",
    ),
    (
        "md",
        """## 5. Profile learning""",
    ),
    (
        "code",
        r"""cProfile.runctx(
    "learn_templates(messages, config=config)",
    globals(),
    locals(),
    filename="/tmp/learn.prof",
)
p = pstats.Stats("/tmp/learn.prof")
p.strip_dirs().sort_stats("cumulative").print_stats(20)
""",
    ),
    (
        "md",
        """## 6. Profile matching""",
    ),
    (
        "code",
        r"""cProfile.runctx(
    "for m in messages: matcher.match(m)",
    globals(),
    locals(),
    filename="/tmp/match.prof",
)
p = pstats.Stats("/tmp/match.prof")
p.strip_dirs().sort_stats("cumulative").print_stats(20)
""",
    ),
    (
        "md",
        """## 7. Sample matches""",
    ),
    (
        "code",
        r"""print("Random sample matches:\n")
for m in random.sample(messages, min(10, len(messages))):
    r = matcher.match(m)
    print(m)
    if r is None:
        print("  -> no match\n")
    else:
        print("  ->", r.template_id, r.entities, "\n")
""",
    ),
    (
        "md",
        """## 8. Full markdown report""",
    ),
    (
        "code",
        r"""report = render_library(library, {**cov, "learn_seconds": learn_seconds, "messages": len(messages)})
print(report)
""",
    ),
    (
        "md",
        """## Next steps

- Replace `CSV_PATH` with your real CSV and rerun all cells.
- Tune `SIM` (similarity threshold) if coverage is too low or templates too noisy.
- Use `LEXICON = "finance"` (or a JSON file) to rename generic slots like `text`
  into domain names like `merchant` / `beneficiary`.
- Persist `model.json` and use `patgen match` or `TemplateMatcher` in production.""",
    ),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    for cell_type, source in CELLS:
        if cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(source))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(source))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "full_pipeline.ipynb"
    nbf.write(build(), str(out))
    print("wrote", out)
