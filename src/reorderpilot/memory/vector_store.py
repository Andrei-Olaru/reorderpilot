"""Semantic memory: a ChromaDB index over customer notes and feedback notes (SPEC section 17, milestone V2).

SQLite (memory/store.py) stays the source of truth. This module is a *derived index*: `sync_customer` copies a
customer's notes into Chroma under stable ids (`note-<id>`, `fb-<id>`), so syncing twice changes nothing and the
whole index can be deleted and rebuilt at any time. The relational suppression rules are NOT moved here: "do not
offer 85123A for 30 days" is a rule and stays deterministic in SQL; the vector store only finds *relevant text*
("the client complained about delivery delays last month") for the live-call copilot.

No leakage: every document stores its simulated creation day as an int (20111101), and `search` only returns
documents created on or before `as_of`, matching the visibility rule of memory/store.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import chromadb

from reorderpilot.memory import store

COLLECTION = "customer_memory"


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory: the text, where it came from, and its distance to the query (lower = closer)."""

    text: str
    source: str        # "note" or "feedback"
    created_at: date
    distance: float


def _day(d: date) -> int:
    """Date as a sortable int (2011-11-01 -> 20111101); Chroma `where` filters compare numbers, not dates."""
    return d.year * 10_000 + d.month * 100 + d.day


class VectorMemory:
    """Chroma collection of customer memories. Pass `embedding_function` to avoid the default ONNX model (tests).

    `path=None` keeps everything in memory; otherwise the index is persisted in that folder (e.g. data/chroma).
    The default embedding function (all-MiniLM-L6-v2, ~80 MB) is downloaded once on first use.
    """

    def __init__(self, path: Path | None = None, embedding_function: object | None = None) -> None:
        settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
        self.client = (chromadb.PersistentClient(path=str(path), settings=settings) if path is not None
                       else chromadb.EphemeralClient(settings=settings))
        kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
        # cosine: distance = 1 - cosine similarity, easier to read than raw L2 distance
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                                               **kwargs)

    def add(self, doc_id: str, customer_id: str, text: str, created_at: date, source: str) -> None:
        """Insert or overwrite one memory (upsert: the same id twice keeps one copy)."""
        self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[{
            "customer_id": customer_id, "source": source, "created_day": _day(created_at),
            "created_at": created_at.isoformat()}])

    def sync_customer(self, db_path: Path, customer_id: str, as_of: date) -> int:
        """Copy the customer's SQLite notes + feedback notes (visible at as_of) into the index; return how many."""
        n = 0
        for note in store.list_customer_notes(db_path, customer_id, as_of):
            self.add(f"note-{note.id}", customer_id, note.note, note.created_at, "note")
            n += 1
        # include_expired: a suppression may have ended, but "they switched supplier in May" is still useful context
        for ev in store.list_feedback_events(db_path, customer_id, as_of, include_expired=True):
            if ev.note:  # only events with free text carry an objection worth retrieving
                text = f"{ev.outcome} {ev.stock_code or ''} ({ev.reason or 'no reason'}): {ev.note}"
                self.add(f"fb-{ev.id}", customer_id, text, ev.created_at, "feedback")
                n += 1
        return n

    def search(self, customer_id: str, query: str, as_of: date, k: int = 3) -> list[MemoryHit]:
        """Top-k memories of this customer closest in meaning to `query`, created on or before as_of."""
        if not query.strip():
            return []
        where = {"$and": [{"customer_id": customer_id}, {"created_day": {"$lte": _day(as_of)}}]}
        res = self.collection.query(query_texts=[query], n_results=k, where=where)
        return [MemoryHit(text=doc, source=meta["source"], created_at=date.fromisoformat(meta["created_at"]),
                          distance=round(float(dist), 4))
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)]

    def count(self) -> int:
        """Number of documents in the index (all customers)."""
        return self.collection.count()
