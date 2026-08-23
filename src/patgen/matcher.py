"""Production phase: match a message against a learned library, fast.

Strategy
--------
* one normalization + tokenization pass produces *canonical text*;
* templates are bucketed by their first literal token, so a message only ever
  competes against the handful of templates that can possibly match it;
* each bucket is compiled into a few large alternation regexes with named
  groups, pushing the inner loop into the C regex engine instead of Python.

Templates are tried most-specific-first, so a generic fallback template never
steals a message from a precise one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from .entities import TEXT
from .learn import prepare
from .normalize import DEFAULT_CONFIG, NormalizeConfig
from .template import Template, TemplateLibrary

BRANCHES_PER_REGEX = 24
WILDCARD_BUCKET = "\x00*"


@dataclass(slots=True)
class MatchResult:
    template_id: str
    label: str
    entities: dict[str, str]
    canonical: str

    def as_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "label": self.label,
            "entities": self.entities,
            "canonical": self.canonical,
        }


class _CompiledGroup:
    __slots__ = ("pattern", "branch_to_template", "slot_names")

    def __init__(self, templates: Sequence[tuple[int, Template]]):
        branches = []
        self.branch_to_template: dict[str, Template] = {}
        self.slot_names: dict[str, list[tuple[str, str, bool]]] = {}
        for index, template in templates:
            branch = f"b{index}"
            body = _prefix_groups(template.to_regex(), index)
            branches.append(f"(?P<{branch}>{body})")
            self.branch_to_template[branch] = template
            self.slot_names[branch] = [
                (f"g{index}_s{slot_index}", slot.key, slot.entity != TEXT)
                for slot_index, slot in enumerate(template.slots)
            ]
        self.pattern = re.compile("|".join(branches), re.UNICODE)

    def match(self, text: str) -> MatchResult | None:
        m = self.pattern.fullmatch(text)
        if m is None:
            return None
        for branch, template in self.branch_to_template.items():
            if m.group(branch) is None:
                continue
            entities = {}
            for group_name, slot_key, tidy in self.slot_names[branch]:
                value = m.group(group_name)
                if value is not None:
                    entities[slot_key] = _tidy(value) if tidy else value
            return MatchResult(template.template_id, template.label, entities, text)
        return None


# Template slots are numbered, e.g. (?P<s0>...); we prefix with the branch index
# so alternation branches do not reuse the same group name.
_BRANCH_SLOT = re.compile(r"\(\?P<s(\d+)>")
# Tokenization spaces punctuation out; typed values read better glued back
# together ("* * * 815" -> "***815", "ر . س" -> "ر.س").
_LOOSE_PUNCT = re.compile(r"\s*([.,:/*#])\s*")


def _tidy(value: str) -> str:
    return _LOOSE_PUNCT.sub(r"\1", value).strip()


def _prefix_groups(pattern: str, index: int) -> str:
    return _BRANCH_SLOT.sub(lambda m: f"(?P<g{index}_s{m.group(1)}>", pattern)


class TemplateMatcher:
    def __init__(
        self,
        library: TemplateLibrary,
        normalize_config: NormalizeConfig = DEFAULT_CONFIG,
    ) -> None:
        self.library = library
        self.normalize_config = normalize_config
        self._buckets: dict[str, list[_CompiledGroup]] = {}
        self._build()

    # -- build ---------------------------------------------------------
    def _build(self) -> None:
        by_bucket: dict[str, list[tuple[int, Template]]] = {}
        for index, template in enumerate(self.library.templates):
            key = template.first_literal or WILDCARD_BUCKET
            by_bucket.setdefault(key, []).append((index, template))
        for key, entries in by_bucket.items():
            entries.sort(key=lambda pair: -pair[1].specificity)
            groups = [
                _CompiledGroup(entries[i : i + BRANCHES_PER_REGEX])
                for i in range(0, len(entries), BRANCHES_PER_REGEX)
            ]
            self._buckets[key] = groups

    # -- serve ----------------------------------------------------------
    def match(self, message: str) -> MatchResult | None:
        prepared = prepare(message, self.normalize_config)
        return self.match_canonical(prepared.canonical)

    def match_canonical(self, canonical: str) -> MatchResult | None:
        if not canonical:
            return None
        head = canonical.split(" ", 1)[0]
        for key in (head, WILDCARD_BUCKET):
            for group in self._buckets.get(key, ()):
                result = group.match(canonical)
                if result is not None:
                    return result
        return None

    def match_many(self, messages: Iterable[str]) -> Iterator[MatchResult | None]:
        for message in messages:
            yield self.match(message)

    def __len__(self) -> int:
        return len(self.library.templates)
