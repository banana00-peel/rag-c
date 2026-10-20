import json
import re
from pathlib import Path
from threading import RLock
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from app.domain.models import ChunkRecord


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in TOKEN_PATTERN.findall(text.lower()):
        if any("\u4e00" <= char <= "\u9fff" for char in part):
            tokens.extend(token for token in jieba.lcut(part) if token.strip())
        else:
            tokens.append(part)
    return tokens


class BM25Store:
    def __init__(self, snapshot_path: str | Path):
        self._snapshot_path = Path(snapshot_path)
        self._lock = RLock()
        self._chunks: list[ChunkRecord] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None

    def rebuild(self, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            self._chunks = list(chunks)
            self._tokenized_corpus = [tokenize(chunk.text) for chunk in self._chunks]
            self._index = BM25Okapi(self._tokenized_corpus) if self._chunks else None
            self._write_snapshot()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        model: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._index is None:
                return []
            scores = self._index.get_scores(tokenize(query))
            ranked_indexes = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            matches: list[dict[str, Any]] = []
            for index in ranked_indexes:
                chunk = self._chunks[index]
                if model and model != "all" and chunk.model != model:
                    continue
                score = float(scores[index])
                if score <= min_score:
                    continue
                matches.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "chunk": chunk,
                        "score": score,
                    }
                )
                if len(matches) >= top_k:
                    break
            return matches

    def _write_snapshot(self) -> None:
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_count": len(self._chunks),
            "chunk_ids": [chunk.id for chunk in self._chunks],
        }
        self._snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

