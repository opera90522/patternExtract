"""Human readable reporting for a learned template library."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .matcher import TemplateMatcher
from .template import Slot, TemplateLibrary


def coverage_report(library: TemplateLibrary, messages: Iterable[str]) -> dict[str, object]:
    matcher = TemplateMatcher(library)
    total = matched = 0
    misses: list[str] = []
    entity_counts: Counter = Counter()
    for message in messages:
        total += 1
        result = matcher.match(message)
        if result is None:
            if len(misses) < 25:
                misses.append(message)
            continue
        matched += 1
        entity_counts.update(result.entities.keys())
    return {
        "messages_scored": total,
        "matched": matched,
        "coverage": matched / total if total else 0.0,
        "unmatched_samples": misses,
        "slot_frequency": dict(entity_counts.most_common()),
    }


def render_library(library: TemplateLibrary, stats: dict[str, object]) -> str:
    lines: list[str] = ["# Template library", ""]
    lines.append(f"- templates: **{len(library)}**")
    for key in ("messages", "messages_scored", "coverage", "learn_seconds"):
        if key in stats:
            value = stats[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            lines.append(f"- {key}: **{value}**")
    lines.append("")
    lines.append("## Templates")
    lines.append("")
    for template in library.templates:
        lines.append(f"### `{template.template_id}` ({template.count} msgs)")
        lines.append("")
        lines.append(f"```\n{template.text}\n```")
        if template.slots:
            lines.append("")
            lines.append("| slot | entity | distinct | examples |")
            lines.append("| --- | --- | --- | --- |")
            for slot in template.slots:
                lines.append(_slot_row(slot))
        if template.examples:
            lines.append("")
            lines.append(f"example: `{template.examples[0]}`")
        lines.append("")
    unmatched = stats.get("unmatched_samples") or []
    if unmatched:
        lines.append("## Unmatched samples")
        lines.append("")
        lines.extend(f"- `{m}`" for m in unmatched)  # type: ignore[union-attr]
    return "\n".join(lines)


def _slot_row(slot: Slot) -> str:
    examples = ", ".join(f"`{e}`" for e in slot.examples[:3])
    return f"| {slot.key} | {slot.entity} | {slot.cardinality} | {examples} |"
