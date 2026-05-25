"""Cycle helpers for graph-based AML rules."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Hashable


def has_short_cycle_before_edge(
    adjacency: dict[Hashable, set[Hashable]],
    sender: Hashable,
    receiver: Hashable,
    max_length: int,
) -> bool:
    """Return true when `receiver` can reach `sender` before adding an edge."""

    if max_length < 2:
        return False
    if sender == receiver:
        return True
    max_hops = max_length - 1
    queue: deque[tuple[Hashable, int]] = deque([(receiver, 0)])
    visited = {receiver}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in adjacency.get(node, set()):
            if neighbor == sender:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return False


def add_edge(adjacency: dict[Hashable, set[Hashable]], sender: Hashable, receiver: Hashable) -> None:
    adjacency.setdefault(sender, set()).add(receiver)


def new_adjacency() -> dict[Hashable, set[Hashable]]:
    return defaultdict(set)

