"""Small connected-component helper for case consolidation."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Iterable


def connected_components(edges: Iterable[tuple[Hashable, Hashable]]) -> list[set[Hashable]]:
    adjacency: dict[Hashable, set[Hashable]] = defaultdict(set)
    nodes: set[Hashable] = set()
    for left, right in edges:
        nodes.add(left)
        nodes.add(right)
        adjacency[left].add(right)
        adjacency[right].add(left)

    components = []
    seen: set[Hashable] = set()
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        component: set[Hashable] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)
    return components

