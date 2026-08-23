"""Learning phase: raw messages -> typed, named template library.

Pipeline
--------
1. normalize (mojibake repair, unicode, Arabic folding)
2. tokenize
3. entity masking            -> bounded vocabulary
4. Drain clustering          -> one cluster per wording family
5. template synthesis        -> literals + typed slots, named from context
6. value pass                -> enums, cardinality, examples, slot widths
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .drain import WILDCARD, Cluster, DrainTree, align_wildcards
from .entities import TEXT, mask_tokens
from .lexicon import load, normalize_lexicon
from .naming import SlotNamer
from .normalize import DEFAULT_CONFIG, NormalizeConfig, normalize
from .template import Literal, Part, Slot, Template, TemplateLibrary, enum_values
from .tokenizer import tokenize

_ENTITY_KEY = re.compile(r"^<([A-Z]+)>$")
DEFAULT_TEXT_SLOT_WIDTH = 6


@dataclass
class LearnConfig:
    sim_threshold: float = 0.5
    depth: int = 4
    max_children: int = 128
    min_support: int = 1
    #: Drop templates that only ever matched one message and are mostly slots.
    drop_degenerate: bool = True
    max_text_slot_tokens: int = DEFAULT_TEXT_SLOT_WIDTH
    #: Re-expand a wildcard when this share of the messages filling it agree on
    #: the same structure, so one truncated message cannot erase the entities
    #: of a whole cluster.
    refine_ratio: float = 0.5
    normalize_config: NormalizeConfig = field(default=DEFAULT_CONFIG)
    lexicon: str | dict[str, str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.lexicon, str):
            self.lexicon = load(self.lexicon)
        if isinstance(self.lexicon, dict):
            self.lexicon = normalize_lexicon(self.lexicon)


@dataclass
class PreparedMessage:
    raw: str
    canonical: str
    keys: list[str]
    values: list[str]
    cluster_id: int = -1


def prepare(text: str, config: NormalizeConfig = DEFAULT_CONFIG) -> PreparedMessage:
    """Normalize + tokenize + mask a single message (shared by learn/serve)."""
    canonical_text = normalize(text, config)
    tokens = tokenize(canonical_text)
    masked = mask_tokens(tokens)
    return PreparedMessage(
        raw=text,
        canonical=" ".join(t.text for t in tokens),
        keys=[m.key for m in masked],
        values=[m.value for m in masked],
    )


class TemplateLearner:
    """Two-pass learner. Pass one clusters, pass two profiles slot values."""

    def __init__(self, config: LearnConfig | None = None) -> None:
        self.config = config or LearnConfig()
        self.tree = DrainTree(
            depth=self.config.depth,
            sim_threshold=self.config.sim_threshold,
            max_children=self.config.max_children,
        )
        self._prepared: list[PreparedMessage] = []

    # -- pass 1 --------------------------------------------------------
    def fit_partial(self, messages: Iterable[str], keep: bool = True) -> None:
        for message in messages:
            if not message or not message.strip():
                continue
            prepared = prepare(message, self.config.normalize_config)
            if not prepared.keys:
                continue
            cluster = self.tree.add(prepared.keys, example=prepared.canonical)
            if keep:
                prepared.cluster_id = cluster.cluster_id
                self._prepared.append(prepared)

    def fit(self, messages: Iterable[str]) -> TemplateLibrary:
        self.fit_partial(messages)
        return self.build()

    # -- pass 2 --------------------------------------------------------
    def build(self) -> TemplateLibrary:
        self._refine_clusters()
        templates: list[Template] = []
        for cluster in self.tree.clusters:
            if cluster.count < self.config.min_support:
                continue
            template = self._synthesize(cluster)
            if template is not None:
                templates.append(template)
        templates = _absorb_specializations(_deduplicate(templates))
        SlotNamer.fit(templates, lexicon=self.config.lexicon).assign_all(templates)
        self._profile_slots(templates)
        templates.sort(key=lambda t: (-t.count, t.template_id))
        return TemplateLibrary(
            templates=templates,
            meta={
                "messages": len(self._prepared),
                "clusters": len(self.tree.clusters),
                "templates": len(templates),
                "sim_threshold": self.config.sim_threshold,
            },
        )

    # -- cluster refinement ---------------------------------------------
    def _refine_clusters(self) -> None:
        """Give wildcards their structure back when the data agrees on one.

        Drain widens a template as soon as one odd message (a truncated tail,
        an extra clause) arrives, and everything the wildcard swallows stops
        being extractable. Each wildcard is therefore re-examined against the
        messages that filled it: a dominant filling is spliced back into the
        template, marked optional when some messages left the wildcard empty.
        """
        if not self._prepared or self.config.refine_ratio <= 0:
            return
        fillings: dict[int, dict[int, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        by_id = {cluster.cluster_id: cluster for cluster in self.tree.clusters}
        for prepared in self._prepared:
            # The training assignment is authoritative: re-matching could pick
            # a sibling cluster and starve this one of its own evidence.
            cluster = by_id.get(prepared.cluster_id)
            if cluster is None or WILDCARD not in cluster.tokens:
                continue
            covered = align_wildcards(cluster.tokens, prepared.keys)
            if covered is None:
                continue
            for index, filler in covered.items():
                fillings[cluster.cluster_id][index][tuple(filler)] += 1
        for cluster in self.tree.clusters:
            per_wildcard = fillings.get(cluster.cluster_id)
            if per_wildcard:
                _apply_refinement(cluster, per_wildcard, self.config.refine_ratio)

    # -- template synthesis --------------------------------------------
    def _synthesize(self, cluster: Cluster) -> Template | None:
        parts: list[Part] = []
        pending_wildcards = 0

        def flush_wildcards(index: int, trailing: bool = False) -> None:
            nonlocal pending_wildcards
            if not pending_wildcards:
                return
            start = index - pending_wildcards
            entity = _wildcard_entity(cluster, start, index)
            width = (
                1
                if entity != TEXT
                else min(
                    self.config.max_text_slot_tokens,
                    max(pending_wildcards, DEFAULT_TEXT_SLOT_WIDTH),
                )
            )
            optional = any(i in cluster.optional for i in range(start, index))
            parts.append(
                Slot(
                    key="",
                    entity=entity,
                    min_tokens=0 if optional else 1,
                    # A trailing free-text slot is unbounded (0): capping it
                    # would silently reject every longer tail at serve time.
                    max_tokens=0
                    if (trailing and entity == TEXT)
                    else max(width, pending_wildcards),
                )
            )
            pending_wildcards = 0

        for i, token in enumerate(cluster.tokens):
            if token == WILDCARD:
                pending_wildcards += 1
                continue
            flush_wildcards(i)
            match = _ENTITY_KEY.match(token)
            if match:
                parts.append(Slot(key="", entity=match.group(1)))
            else:
                parts.append(Literal(token))
        flush_wildcards(len(cluster.tokens), trailing=True)

        if not parts:
            return None
        if self.config.drop_degenerate and _is_degenerate(parts, cluster.count):
            return None

        template = Template(
            template_id=f"t{cluster.cluster_id:05d}",
            parts=parts,
            count=cluster.count,
            examples=list(cluster.examples),
        )
        template.label = _label(template)
        return template

    # -- slot profiling -------------------------------------------------
    def _profile_slots(self, templates: Sequence[Template]) -> None:
        if not self._prepared:
            return
        compiled = [(t, t.compile()) for t in templates]
        # Prefer the most specific template when several can claim a message.
        compiled.sort(key=lambda pair: -pair[0].specificity)
        observed: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        for prepared in self._prepared:
            for template, pattern in compiled:
                match = pattern.fullmatch(prepared.canonical)
                if match is None:
                    continue
                for key, value in match.groupdict().items():
                    if value is not None:
                        observed[template.template_id][key][value] += 1
                break
        for template in templates:
            stats = observed.get(template.template_id)
            if not stats:
                continue
            for slot in template.slots:
                counter = stats.get(slot.key)
                if not counter:
                    continue
                slot.cardinality = len(counter)
                slot.examples = [v for v, _ in counter.most_common(5)]
                if slot.entity == TEXT:
                    if slot.max_tokens:
                        slot.max_tokens = max(
                            1, max(value.count(" ") + 1 for value in counter)
                        )
                    slot.values = enum_values(list(counter), sum(counter.values()))


def _absorb_specializations(templates: list[Template]) -> list[Template]:
    """Drop templates that a more popular one already covers.

    Clustering sometimes freezes a value into a literal (one merchant name
    seen a handful of times), leaving a template strictly narrower than a
    sibling. Its traffic is handed over to the general template.
    """
    ordered = sorted(templates, key=lambda t: (-t.count, t.template_id))
    compiled = [(t, t.compile()) for t in ordered]
    dropped: set[str] = set()
    for template, _ in reversed(compiled):
        if not template.examples:
            continue
        for other, pattern in compiled:
            if other.template_id == template.template_id:
                break
            if other.template_id in dropped:
                continue
            if all(pattern.fullmatch(example) for example in template.examples):
                other.count += template.count
                dropped.add(template.template_id)
                break
    return [t for t in ordered if t.template_id not in dropped]


def _dominant_affix(counter: Counter, ratio: float, from_end: bool) -> list[str]:
    """Longest prefix/suffix shared by at least ``ratio`` of the fillings."""
    total = sum(counter.values())
    if not total:
        return []
    best: Sequence[str] = ()
    longest = max((len(filler) for filler in counter), default=0)
    for length in range(1, longest + 1):
        sub: Counter = Counter()
        for filler, hits in counter.items():
            if len(filler) >= length:
                sub[filler[-length:] if from_end else filler[:length]] += hits
        if not sub:
            break
        candidate, hits = sub.most_common(1)[0]
        if hits / total < ratio:
            break
        best = candidate
    return list(best)


def _refine_wildcard(counter: Counter, ratio: float) -> tuple[list[str], bool]:
    """Rebuild one wildcard from what filled it.

    Returns the replacement tokens (at most one wildcard left, for the part
    that really varies) and whether the region can be empty.
    """
    total = sum(counter.values())
    empty = counter.get((), 0)
    filler, hits = max(counter.items(), key=lambda kv: kv[1])
    if filler and hits / total >= ratio:
        return list(filler), hits < total

    suffix = _dominant_affix(counter, ratio, from_end=True)
    # The prefix is then measured only on the fillings that carry the suffix,
    # otherwise a handful of truncated messages shrink both affixes at once.
    tail = tuple(suffix)
    heads: Counter = Counter()
    for filler, hits in counter.items():
        if len(filler) > len(tail) and filler[len(filler) - len(tail) :] == tail:
            heads[filler[: len(filler) - len(tail)]] += hits
    prefix = _dominant_affix(heads, ratio, from_end=False) if heads else []
    shortest_head = min((len(f) for f in heads if f), default=0)
    while prefix and len(prefix) >= shortest_head:
        prefix.pop()
    fixed = len(prefix) + len(suffix)
    can_be_empty = bool(empty) or any(len(f) <= fixed for f in counter)
    if can_be_empty:
        # If a clause is optional, do not freeze its affixes as required text;
        # keeping the wildcard optional preserves coverage for the empty case.
        return [WILDCARD], can_be_empty
    if not prefix and not suffix:
        return [WILDCARD], can_be_empty
    return prefix + [WILDCARD] + suffix, can_be_empty


def _apply_refinement(
    cluster: Cluster, fillings: dict[int, Counter], ratio: float
) -> None:
    tokens: list[str] = []
    optional: set[int] = set()
    slot_values: dict[int, dict[str, int]] = {}
    for index, token in enumerate(cluster.tokens):
        if token != WILDCARD:
            tokens.append(token)
            continue
        counter = fillings.get(index)
        if not counter:
            if index in cluster.optional:
                optional.add(len(tokens))
            tokens.append(WILDCARD)
            continue
        replacement, can_be_empty = _refine_wildcard(counter, ratio)
        start = len(tokens)
        tokens.extend(replacement)
        if can_be_empty or index in cluster.optional:
            optional.update(range(start, len(tokens)))
        if WILDCARD in replacement:
            index_of_wildcard = start + replacement.index(WILDCARD)
            slot_values[index_of_wildcard] = _wildcard_fillers(counter, replacement)
    cluster.slot_values = slot_values
    cluster.optional = optional
    cluster.tokens = tokens


def _wildcard_fillers(counter: Counter, replacement: Sequence[str]) -> dict[str, int]:
    """Tokens still covered by the wildcard once the affixes are spliced back."""
    position = replacement.index(WILDCARD)
    tail = len(replacement) - position - 1
    values: dict[str, int] = {}
    for filler, hits in counter.items():
        for token in filler[position : len(filler) - tail]:
            values[token] = values.get(token, 0) + hits
    return values


def _wildcard_entity(cluster: Cluster, start: int, end: int) -> str:
    """Decide what a wildcard run really holds, from the values Drain absorbed."""
    entities = set()
    saw_literal = False
    for index in range(start, end):
        for value in cluster.slot_values.get(index, {}):
            match = _ENTITY_KEY.match(value)
            if match:
                entities.add(match.group(1))
            else:
                saw_literal = True
    if len(entities) == 1 and not saw_literal and end - start == 1:
        return entities.pop()
    return TEXT


def _is_degenerate(parts: Sequence[Part], count: int) -> bool:
    literals = sum(1 for p in parts if isinstance(p, Literal))
    alnum_literals = sum(
        1 for p in parts if isinstance(p, Literal) and any(c.isalnum() for c in p.text)
    )
    if alnum_literals == 0:
        return True
    return literals < 2 and count <= 1


def _deduplicate(templates: Sequence[Template]) -> list[Template]:
    """Merge templates that generalized to the same shape in different leaves."""
    by_signature: dict[str, Template] = {}
    for template in templates:
        existing = by_signature.get(template.signature)
        if existing is None:
            by_signature[template.signature] = template
            continue
        existing.count += template.count
        for slot, other in zip(existing.slots, template.slots, strict=False):
            slot.max_tokens = max(slot.max_tokens, other.max_tokens)
        for example in template.examples:
            if len(existing.examples) < 3:
                existing.examples.append(example)
    return list(by_signature.values())


def _label(template: Template) -> str:
    words = [
        p.text
        for p in template.parts
        if isinstance(p, Literal) and any(c.isalnum() for c in p.text)
    ]
    return " ".join(words[:4])


def learn_templates(
    messages: Iterable[str], config: LearnConfig | None = None
) -> TemplateLibrary:
    return TemplateLearner(config).fit(messages)
