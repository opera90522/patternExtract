"""Template representation, regex compilation and (de)serialization."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .entities import ENTITY_REGEX, TEXT

MAX_ENUM_VALUES = 24


@dataclass(frozen=True, slots=True)
class Literal:
    text: str

    def to_regex(self) -> str:
        return re.escape(self.text)


@dataclass(slots=True)
class Slot:
    key: str  # unique within the template, e.g. "amount", "amount_2"
    entity: str
    min_tokens: int = 1
    max_tokens: int = 1
    values: list[str] | None = None  # closed set, when one was observed
    examples: list[str] = field(default_factory=list)
    cardinality: int = 0

    @property
    def is_enum(self) -> bool:
        return bool(self.values)

    @property
    def optional(self) -> bool:
        return self.min_tokens == 0

    def value_regex(self) -> str:
        if self.values:
            return "|".join(sorted((re.escape(v) for v in self.values), key=len, reverse=True))
        base = ENTITY_REGEX.get(self.entity, ENTITY_REGEX[TEXT])
        if self.entity == TEXT:
            if self.max_tokens == 0:  # unbounded, only used for trailing slots
                return r"\S+(?: \S+)*"
            if self.max_tokens > 1:
                return rf"\S+(?: \S+){{0,{self.max_tokens - 1}}}"
        return base

    def to_regex(self) -> str:
        return f"(?P<{self.key}>{self.value_regex()})"


Part = Literal | Slot


@dataclass(slots=True)
class Template:
    template_id: str
    parts: list[Part]
    count: int = 0
    examples: list[str] = field(default_factory=list)
    label: str = ""

    # -- rendering ----------------------------------------------------
    @property
    def text(self) -> str:
        out = []
        for part in self.parts:
            if isinstance(part, Literal):
                out.append(part.text)
            else:
                suffix = "?" if part.optional else ""
                out.append(f"<{part.entity}:{part.key}>{suffix}")
        return " ".join(out)

    @property
    def slots(self) -> list[Slot]:
        return [p for p in self.parts if isinstance(p, Slot)]

    @property
    def first_literal(self) -> str | None:
        for part in self.parts:
            if isinstance(part, Literal):
                return part.text
            return None
        return None

    @property
    def min_tokens(self) -> int:
        return sum(1 if isinstance(p, Literal) else p.min_tokens for p in self.parts)

    @property
    def signature(self) -> str:
        """Identity used to deduplicate templates that generalized alike."""
        return " ".join(
            p.text if isinstance(p, Literal) else f"<{p.entity}{'?' if p.optional else ''}>"
            for p in self.parts
        )

    @property
    def max_tokens(self) -> int:
        return sum(1 if isinstance(p, Literal) else p.max_tokens for p in self.parts)

    @property
    def specificity(self) -> int:
        """Literal token count; used to prefer precise templates when several match."""
        return sum(1 for p in self.parts if isinstance(p, Literal))

    def to_regex(self) -> str:
        """Regex over canonical text (tokens joined by single spaces).

        Separators are ``\\s*`` around punctuation-only literals so that
        ``100.00 sar`` and ``100.00sar`` land on the same template.
        """
        chunks: list[str] = []
        for i, part in enumerate(self.parts):
            separator = (
                ""
                if i == 0
                else (r"\s*" if _optional_space(self.parts[i - 1], part) else r"\s+")
            )
            if isinstance(part, Slot) and part.optional:
                # The separator lives inside the optional group, otherwise a
                # message without the trailing clause keeps a dangling space.
                chunks.append(f"(?:{separator}{part.to_regex()})?")
            else:
                chunks.append(separator + part.to_regex())
        return "".join(chunks)

    def compile(self) -> re.Pattern:
        return re.compile(self.to_regex(), re.UNICODE)

    # -- serialization ------------------------------------------------
    def to_dict(self) -> dict:
        parts = []
        for part in self.parts:
            if isinstance(part, Literal):
                parts.append({"type": "literal", "text": part.text})
            else:
                parts.append(
                    {
                        "type": "slot",
                        "key": part.key,
                        "entity": part.entity,
                        "min_tokens": part.min_tokens,
                        "max_tokens": part.max_tokens,
                        "values": part.values,
                        "examples": part.examples,
                        "cardinality": part.cardinality,
                    }
                )
        return {
            "template_id": self.template_id,
            "label": self.label,
            "count": self.count,
            "text": self.text,
            "regex": self.to_regex(),
            "examples": self.examples,
            "parts": parts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Template:
        parts: list[Part] = []
        for raw in data["parts"]:
            if raw["type"] == "literal":
                parts.append(Literal(raw["text"]))
            else:
                parts.append(
                    Slot(
                        key=raw["key"],
                        entity=raw["entity"],
                        min_tokens=raw.get("min_tokens", 1),
                        max_tokens=raw.get("max_tokens", 1),
                        values=raw.get("values"),
                        examples=raw.get("examples", []),
                        cardinality=raw.get("cardinality", 0),
                    )
                )
        return cls(
            template_id=data["template_id"],
            parts=parts,
            count=data.get("count", 0),
            examples=data.get("examples", []),
            label=data.get("label", ""),
        )


def _optional_space(left: Part, right: Part) -> bool:
    def punct(part: Part) -> bool:
        return isinstance(part, Literal) and not any(ch.isalnum() for ch in part.text)

    return punct(left) or punct(right)


@dataclass
class TemplateLibrary:
    templates: list[Template] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    def __iter__(self) -> Iterable[Template]:
        return iter(self.templates)

    def __len__(self) -> int:
        return len(self.templates)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "meta": self.meta,
                "templates": [t.to_dict() for t in self.templates],
            },
            ensure_ascii=False,
            indent=indent,
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> TemplateLibrary:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            templates=[Template.from_dict(t) for t in data["templates"]],
            meta=data.get("meta", {}),
        )


def enum_values(values: Sequence[str], total: int) -> list[str] | None:
    """Return a closed value set when the slot really looks categorical."""
    distinct = sorted(set(values))
    if not distinct or len(distinct) > MAX_ENUM_VALUES:
        return None
    if total and len(distinct) > max(4, total // 4):
        return None
    if any(" " in v for v in distinct):
        return None
    return distinct
