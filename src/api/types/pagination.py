"""Relay-style cursor pagination for GraphQL connections.

Provides encode/decode cursor helpers and a generic paginate_df()
that slices a DataFrame into a Connection with edges + pageInfo + totalCount.
"""

from __future__ import annotations

import base64
from typing import Any, Callable, Generic, TypeVar

import strawberry

T = TypeVar("T")


def encode_cursor(offset: int) -> str:
    """Encode a 0-based offset into an opaque base64 cursor."""
    return base64.b64encode(f"cursor:{offset}".encode()).decode()


def decode_cursor(cursor: str) -> int | None:
    """Decode an opaque cursor back to a 0-based offset. Returns None on failure."""
    try:
        decoded = base64.b64decode(cursor).decode()
        if decoded.startswith("cursor:"):
            return int(decoded.split(":")[1])
    except (ValueError, IndexError):
        pass
    return None


@strawberry.type
class PageInfo:
    has_next_page: bool
    has_previous_page: bool
    start_cursor: str | None = None
    end_cursor: str | None = None


@strawberry.type
class Connection(Generic[T]):
    edges: list["Edge[T]"]
    page_info: PageInfo
    total_count: int


@strawberry.type
class Edge(Generic[T]):
    cursor: str
    node: T


def paginate_df(
    df: Any,
    to_node: Callable[[Any], T],
    first: int = 25,
    after: str | None = None,
) -> Connection[T]:
    """Paginate a DataFrame into a Relay Connection.

    Args:
        df: Filtered pandas DataFrame (already subset to user's criteria).
        to_node: Callable that converts a DataFrame row (Series) to a Strawberry type.
        first: Number of items to return (max 100).
        after: Opaque cursor for the starting position.

    Returns:
        Connection with edges, pageInfo, and totalCount.
    """
    first = max(1, min(first, 100))
    total_count = len(df)

    start = 0
    if after:
        decoded = decode_cursor(after)
        if decoded is not None:
            start = decoded + 1

    end = start + first
    sliced = df.iloc[start:end]

    edges = []
    for idx_offset, (iloc_pos, row) in enumerate(sliced.iterrows()):
        cursor = encode_cursor(start + idx_offset)
        edges.append(Edge(cursor=cursor, node=to_node(row)))

    has_next_page = end < total_count
    has_previous_page = start > 0

    return Connection(
        edges=edges,
        page_info=PageInfo(
            has_next_page=has_next_page,
            has_previous_page=has_previous_page,
            start_cursor=edges[0].cursor if edges else None,
            end_cursor=edges[-1].cursor if edges else None,
        ),
        total_count=total_count,
    )
