from typing import TypedDict


class QueryMatch(TypedDict):
    chunk: str
    path: str
    chunk_n: int
    score: float
    document: dict


class QueryResponse(TypedDict):
    matches: list[QueryMatch]
