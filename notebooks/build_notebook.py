"""Generate notebooks/approach.ipynb.

Keeping the notebook generated from a plain script means it stays reviewable
in diffs; run this file and re-execute the notebook to refresh the outputs.
"""

from __future__ import annotations

import pathlib

import nbformat as nbf

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# patgen — mining templates from messy bilingual SMS

This notebook walks the whole approach on a synthetic corpus that mixes
finance, e-commerce shipping, travel bookings, support, 2FA and log alerts:
Arabic + English, Arabic-Indic digits, tashkeel, bidi marks, cp1252 mojibake
in the middle of otherwise clean text, and truncated tails.

1. the data and what is wrong with it
2. normalization + mojibake repair
3. tokenization + entity masking (the step that makes Drain work here)
4. Drain-style clustering and wildcard refinement
5. the learned library: templates, typed slots, coverage
6. production matching + throughput
7. what stays unmatched, and tuning""",
    ),
    (
        "code",
        """import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

from patgen import (  # noqa: E402
    DrainTree,
    LearnConfig,
    TemplateMatcher,
    learn_templates,
    normalize,
    prepare,
    repair_mojibake,
    tokenize,
)
from patgen.entities import mask_tokens  # noqa: E402
from patgen.io_csv import read_texts  # noqa: E402
from patgen.report import coverage_report  # noqa: E402

CSV = Path.cwd().parent / "examples" / "sms_sample.csv"
messages = list(read_texts([CSV]))
len(messages), messages[0]""",
    ),
    ("md", "## 1. What the raw data looks like"),
    (
        "code",
        """random.seed(7)
for m in random.sample(messages, 8):
    print(repr(m))""",
    ),
    (
        "md",
        """Three separate problems in one column:

* **two scripts** — Arabic and English templates, often mixed inside a message;
* **encoding damage** — `Ø±.Ø³` is `ر.س` that went through UTF-8 → cp1252,
  and it is usually only *part* of the message, so a whole-string round trip
  cannot fix it;
* **noise** — Arabic-Indic digits, tashkeel, bidi marks, double spaces,
  truncated tails.""",
    ),
    (
        "code",
        """broken = [m for m in messages if re.search("[ÂÃØÙÚÛ][\\u0080-\\u00ff\\u0152-\\u0178]", m)]
print(f"{len(broken)}/{len(messages)} messages carry mojibake\\n")
for m in broken[:3]:
    print("raw :", m)
    print("fixed:", repair_mojibake(m), "\\n")""",
    ),
    ("md", "## 2. Normalization"),
    (
        "code",
        """samples = [
    "‏سحب نقدي ٦٩,٩٥١.٧٠ ر.س من الحساب ***815‎",
    "Bill payment Ø±.Ø³ 37,955.74 to ACME",
    "رَصيدُك الحالي ٢٥٠٫٥٠ ريال",
    "عملية شراء لدى صيدلية النهدي",
]
for s in samples:
    print(f"{s!r}\\n  -> {normalize(s)!r}")""",
    ),
    (
        "md",
        """Folding (`أإآ→ا`, `ة→ه`, `ى→ي`), digit conversion and mojibake repair all
happen *before* clustering, so the same sentence written four different ways
collapses onto one template instead of four.""",
    ),
    ("md", "## 3. Tokenization + entity masking"),
    (
        "code",
        """msg = "Purchase of ر.س 4,467.79 at starbucks on card ending ****5744 on 29/07/2024 02:48. Available balance ر.س 110,579.86"
prepared = prepare(msg)
print("canonical:", prepared.canonical, "\\n")
for m in mask_tokens(tokenize(prepared.canonical)):
    flag = "  <-- entity" if m.is_entity else ""
    print(f"{m.key:<12} {m.value}{flag}")""",
    ),
    (
        "md",
        """This is the key difference from vanilla Drain3. Values are replaced by
**typed** placeholders before clustering, so amounts, dates and card tails
never split a cluster, and the placeholder type is carried into the template
(a plain `<*>` would lose it).""",
    ),
    ("md", "## 4. Clustering, and why wildcards get refined"),
    (
        "code",
        """tree = DrainTree()
for m in messages[:2000]:
    tree.add(prepare(m).keys)
print(f"{len(tree.clusters)} raw clusters from 2000 messages")
for c in sorted(tree.clusters, key=lambda c: -c.count)[:6]:
    print(f"{c.count:>5}  {' '.join(c.tokens)}")""",
    ),
    (
        "md",
        """A single truncated message widens a cluster into `<*>`, swallowing fields
that were perfectly extractable. The learner replays every wildcard against the
messages that filled it and splices back the dominant filling — or the dominant
prefix/suffix around the part that genuinely varies — marking it optional when
some messages left it empty.""",
    ),
    (
        "code",
        """t0 = time.perf_counter()
library = learn_templates(messages)
print(f"{len(messages)} messages -> {len(library.templates)} templates in {time.perf_counter()-t0:.2f}s")
for t in library.templates[:10]:
    print(f"{t.count:>5}  {t.text}")""",
    ),
    ("md", "## 5. Typed slots per template"),
    (
        "code",
        """t = library.templates[1]
print(t.text, "\\n")
for slot in t.slots:
    print(f"{slot.key:<12} {slot.entity:<9} distinct={slot.cardinality:<5} e.g. {slot.examples[:3]}")""",
    ),
    (
        "code",
        """counts = Counter(slot.key for tpl in library.templates for slot in tpl.slots)
counts.most_common(15)""",
    ),
    ("md", "## 6. Production matching"),
    (
        "code",
        """matcher = TemplateMatcher(library)
for m in random.sample(messages, 5):
    r = matcher.match(m)
    print(m)
    if r is None:
        print("   -> no match\\n")
    else:
        print("   ->", r.template_id, r.entities, "\\n")""",
    ),
    (
        "code",
        """batch = messages * 3
t0 = time.perf_counter()
hits = sum(1 for r in matcher.match_many(batch) if r)
elapsed = time.perf_counter() - t0
print(f"{len(batch)} messages in {elapsed:.2f}s = {len(batch)/elapsed:,.0f} msg/s, {hits/len(batch):.1%} matched")""",
    ),
    (
        "md",
        """One normalization pass, then a bucket lookup on the first literal token and
a handful of alternation regexes with named groups — the inner loop runs in the
C regex engine, not in Python.""",
    ),
    ("md", "## 7. What is left unmatched, and tuning"),
    (
        "code",
        """stats = coverage_report(library, messages)
print(f"coverage {stats['coverage']:.1%}")
for m in stats["unmatched_samples"][:8]:
    print(" ", m)""",
    ),
    (
        "md",
        """The residue is genuinely damaged traffic: messages cut mid-word by the SMS
gateway, or mojibake where one byte of the UTF-8 pair was dropped and nothing
can decode it back. Those are the rows worth alerting on, not templating.""",
    ),
    (
        "code",
        """for sim in (0.3, 0.4, 0.5, 0.6, 0.7):
    lib = learn_templates(messages, LearnConfig(sim_threshold=sim))
    cov = coverage_report(lib, messages)["coverage"]
    print(f"sim={sim:<4} templates={len(lib.templates):<5} coverage={cov:.1%}")""",
    ),
    (
        "md",
        """Higher similarity = more, tighter templates; lower = fewer, more generic
ones. `--min-support` then trims the long tail before the model is saved with
`library.save("model.json")` and loaded by the serving process.""",
    ),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    path = pathlib.Path(__file__).with_name("approach.ipynb")
    nbf.write(build(), path)
    print(f"wrote {path}")
