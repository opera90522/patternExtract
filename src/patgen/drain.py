"""Drain-style streaming log clustering, adapted to bilingual SMS.

Differences from the classic Drain / drain3 implementation:

* the input is already entity-masked, so numbers never explode the tree and
  the similarity threshold only has to deal with genuine wording variation;
* the first tree level buckets on token count *ranges* rather than the exact
  count, so an optional trailing clause ("thank you", "شكرا لك") or a
  truncated tail keeps the message in its family;
* clusters absorb length drift by collapsing the differing region into a
  wildcard, remembering whether that region can be empty — that is what later
  becomes an optional slot;
* wildcard positions keep a bounded sample of the values they absorbed, which
  the learning phase turns into typed, named slots.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

WILDCARD = "<*>"


@dataclass
class Cluster:
    cluster_id: int
    tokens: list[str]
    count: int = 0
    #: position -> distinct values seen at a wildcard position (bounded).
    slot_values: dict[int, dict[str, int]] = field(default_factory=dict)
    #: wildcard positions that were absent from at least one message.
    optional: set[int] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)

    @property
    def template(self) -> str:
        return " ".join(self.tokens)


class DrainTree:
    def __init__(
        self,
        depth: int = 4,
        sim_threshold: float = 0.5,
        max_children: int = 128,
        max_slot_values: int = 64,
        max_examples: int = 3,
        length_bucket: int = 4,
        length_tolerance: float = 0.15,
    ) -> None:
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.max_slot_values = max_slot_values
        self.max_examples = max_examples
        self.length_bucket = length_bucket
        self.length_tolerance = length_tolerance
        self._root: dict[object, dict] = {}
        self.clusters: list[Cluster] = []

    # -- tree helpers -------------------------------------------------
    def _prefix_key(self, token: str) -> str:
        """Tokens that are unstable by nature never become tree edges."""
        if token.startswith("<") and token.endswith(">"):
            return WILDCARD
        if any(ch.isdigit() for ch in token):
            return WILDCARD
        return token

    def _leaf(self, tokens: Sequence[str], create: bool) -> list[Cluster] | None:
        node = self._root
        key: object = min(len(tokens) // self.length_bucket, 32)
        for level in range(self.depth - 1):
            child = node.get(key)
            if child is None:
                if not create:
                    child = node.get(WILDCARD)
                    if child is None:
                        return None
                else:
                    if level > 0 and len(node) >= self.max_children and key != WILDCARD:
                        key = WILDCARD
                        child = node.get(key)
                    if child is None:
                        child = [] if level == self.depth - 2 else {}
                        node[key] = child
            if level == self.depth - 2:
                return child  # type: ignore[return-value]
            node = child  # type: ignore[assignment]
            key = self._prefix_key(tokens[level]) if level < len(tokens) else "<EMPTY>"
        return None

    # -- similarity ---------------------------------------------------
    @staticmethod
    def _similarity(template: Sequence[str], tokens: Sequence[str]) -> float:
        """Position-wise agreement, tolerant to a length difference.

        Extra tokens on either side count as mismatches, which lets the
        threshold decide whether an optional clause is still the same template.
        """
        longest = max(len(template), len(tokens))
        if longest == 0:
            return 1.0
        shared = 0
        for a, b in zip(template, tokens, strict=False):
            if a == b or a == WILDCARD:
                shared += 1
        return shared / longest

    def _best(self, clusters: Sequence[Cluster], tokens: Sequence[str]):
        best: Cluster | None = None
        best_sim = -1.0
        for cluster in clusters:
            sim = self._similarity(cluster.tokens, tokens)
            if sim > best_sim:
                best, best_sim = cluster, sim
        return best, best_sim

    # -- public API ---------------------------------------------------
    def add(self, tokens: Sequence[str], example: str = "") -> Cluster:
        leaf = self._leaf(tokens, create=True)
        assert leaf is not None
        cluster, sim = self._best(leaf, tokens)
        if cluster is None or sim < self.sim_threshold or not self._can_absorb(cluster, tokens):
            cluster = Cluster(len(self.clusters), list(tokens))
            self.clusters.append(cluster)
            leaf.append(cluster)
        else:
            self._merge(cluster, tokens)
        cluster.count += 1
        if example and len(cluster.examples) < self.max_examples:
            cluster.examples.append(example)
        return cluster

    def _can_absorb(self, cluster: Cluster, tokens: Sequence[str]) -> bool:
        """Reject length drift that would destroy structure.

        Absorbing an optional trailing clause is desirable; absorbing a
        truncated message would replace a whole structured tail (account,
        date, balance) with one opaque wildcard, so it gets its own cluster.
        """
        longest = max(len(cluster.tokens), len(tokens))
        drift = abs(len(cluster.tokens) - len(tokens))
        return drift <= max(2, int(longest * self.length_tolerance))

    def match(self, tokens: Sequence[str]) -> Cluster | None:
        leaf = self._leaf(tokens, create=False)
        if not leaf:
            return None
        cluster, sim = self._best(leaf, tokens)
        return cluster if cluster is not None and sim >= self.sim_threshold else None

    # -- merging -------------------------------------------------------
    def _merge(self, cluster: Cluster, tokens: Sequence[str]) -> None:
        if len(cluster.tokens) != len(tokens):
            self._collapse(cluster, tokens)
            return
        template = cluster.tokens
        for i, (a, b) in enumerate(zip(template, tokens, strict=False)):
            if a == b:
                continue
            if a != WILDCARD:
                template[i] = WILDCARD
                self._record(cluster, i, a)
            self._record(cluster, i, b)

    def _collapse(self, cluster: Cluster, tokens: Sequence[str]) -> None:
        """Fold a differently sized message into the cluster template.

        The shared prefix and suffix stay literal; the differing middle becomes
        one wildcard. When one side has nothing in that middle the wildcard is
        marked optional, which is how "... thank you" and "..." end up on the
        same template instead of two.
        """
        old = cluster.tokens
        prefix, suffix = common_affixes(old, tokens)
        wildcard_index = prefix
        absorbed: dict[str, int] = {}
        for source in (old, tokens):
            for value in source[prefix : len(source) - suffix]:
                if value != WILDCARD:
                    absorbed[value] = absorbed.get(value, 0) + 1

        cluster.slot_values = _remap(
            cluster.slot_values,
            prefix,
            suffix,
            len(old),
            wildcard_index,
            absorbed,
            self.max_slot_values,
        )
        cluster.optional = {
            index
            for index in (
                _remap_index(i, prefix, suffix, len(old), wildcard_index)
                for i in cluster.optional
            )
            if index >= 0
        }
        if prefix + suffix >= min(len(old), len(tokens)):
            cluster.optional.add(wildcard_index)
        cluster.tokens = old[:prefix] + [WILDCARD] + old[len(old) - suffix :]

    def _record(self, cluster: Cluster, index: int, value: str) -> None:
        values = cluster.slot_values.setdefault(index, {})
        if value in values:
            values[value] += 1
        elif len(values) < self.max_slot_values:
            values[value] = 1


def align_wildcards(
    template: Sequence[str], tokens: Sequence[str]
) -> dict[int, list[str]] | None:
    """Map every wildcard position of ``template`` to the tokens it covers.

    Returns ``None`` when the token sequence cannot be laid over the template.
    Anchoring is greedy on the next literal, which is enough here because
    templates are short and their literals are highly discriminative.
    """
    covered: dict[int, list[str]] = {}
    i = 0
    position = 0
    size = len(template)
    while i < size:
        token = template[i]
        if token != WILDCARD:
            if position >= len(tokens) or tokens[position] != token:
                return None
            i += 1
            position += 1
            continue
        j = i + 1
        while j < size and template[j] == WILDCARD:
            j += 1
        if j == size:
            covered[i] = list(tokens[position:])
            return covered
        anchor = template[j]
        needed = sum(1 for t in template[j:] if t != WILDCARD)
        limit = len(tokens) - needed
        found = -1
        for k in range(position, limit + 1):
            if tokens[k] == anchor:
                found = k
                break
        if found < 0:
            return None
        covered[i] = list(tokens[position:found])
        position = found
        i = j
    return covered if position == len(tokens) else None


def common_affixes(a: Sequence[str], b: Sequence[str]) -> tuple[int, int]:
    """Length of the shared prefix and suffix of two token sequences."""
    shortest = min(len(a), len(b))
    prefix = 0
    while prefix < shortest and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < shortest - prefix and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]:
        suffix += 1
    return prefix, suffix


def _remap_index(
    index: int, prefix: int, suffix: int, old_length: int, wildcard_index: int
) -> int:
    if index < prefix:
        return index
    if index >= old_length - suffix:
        return wildcard_index + 1 + (index - (old_length - suffix))
    return -1  # swallowed by the wildcard


def _remap(
    values: dict[int, dict[str, int]],
    prefix: int,
    suffix: int,
    old_length: int,
    wildcard_index: int,
    absorbed: dict[str, int],
    max_values: int,
) -> dict[int, dict[str, int]]:
    """Shift recorded slot values to the positions of the collapsed template."""
    out: dict[int, dict[str, int]] = {}

    def add(index: int, value: str, count: int) -> None:
        merged = out.setdefault(index, {})
        if value in merged:
            merged[value] += count
        elif len(merged) < max_values:
            merged[value] = count

    for index, counts in values.items():
        new_index = _remap_index(index, prefix, suffix, old_length, wildcard_index)
        if new_index < 0:
            new_index = wildcard_index
        for value, count in counts.items():
            add(new_index, value, count)
    for value, count in absorbed.items():
        add(wildcard_index, value, count)
    return out
