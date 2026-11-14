"""Vector database with semantic search. Replaces src/lib/vectorDb.ts (which faked 64-number "embeddings").

Store:      Chroma (open source, persisted on disk, cosine distance), through LangChain's `Chroma` wrapper.
Embeddings: 1. a neural open-source model served by Ollama (default nomic-embed-text) when it is running;
            2. otherwise latent semantic analysis (TF-IDF + SVD, scikit-learn) fitted on the policy text plus a short
               glossary of how people describe each topic. LSA is real, dense and offline, but it is weaker than a neural model:
               it only knows the words in its own corpus. `embedder_name()` says which one is in use.
Retrieval:  `semantic_clauses()` (vectors only), and `hybrid_clauses()` = BM25 + vectors merged by reciprocal rank fusion.
Collections are named after a fingerprint of (embedder, corpus), so a changed policy or model never reads stale vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .policy_kb import CLAUSES, DATA, TERM_MAP, ClauseRetriever, clause_text, tokenize

DIR = Path(os.getenv("VECTOR_DB_DIR", Path(__file__).parent.parent.parent / ".vectordb"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_CONFIG = json.loads((DATA.parent / "config.json").read_text())
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", _CONFIG["models"]["embedding"])
GLOSSARY = json.loads((DATA / "glossary.json").read_text())


class LsaEmbeddings(Embeddings):
    """Dense semantic vectors from TF-IDF + truncated SVD. Fitted once on the corpus, then used for documents and queries alike."""

    def __init__(self, corpus: list[str], dims: int = 40) -> None:
        self.vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
                                 ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
        x = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=min(dims, x.shape[0] - 1, x.shape[1] - 1), random_state=7).fit(x)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        v = self.svd.transform(self.vec.transform(texts))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def corpus_texts() -> list[str]:
    return [clause_text(c) for c in CLAUSES] + [g["text"] for g in GLOSSARY] + [k + " " + v["normalized"] for k, v in TERM_MAP.items()]


@lru_cache(maxsize=1)
def _ollama_embeddings():
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).json()
        if not any(m["name"].split(":")[0] == EMBED_MODEL for m in tags.get("models", [])):
            return None
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    except Exception:
        return None


@lru_cache(maxsize=1)
def embedder() -> tuple[Embeddings, str]:
    o = _ollama_embeddings()
    return (o, f"ollama:{EMBED_MODEL}") if o else (LsaEmbeddings(corpus_texts()), "lsa-tfidf-svd (offline)")


def embedder_name() -> str:
    return embedder()[1]


@lru_cache(maxsize=1)
def _glossary_embeddings() -> np.ndarray:
    return np.asarray(embedder()[0].embed_documents([item["text"] for item in GLOSSARY]))


def _fingerprint(extra: str = "") -> str:
    return hashlib.sha256((embedder_name() + "|" + "|".join(corpus_texts()) + extra).encode()).hexdigest()[:10]


_LOCK = threading.RLock()


def _store(prefix: str, docs: list[Document], extra: str = "") -> Chroma:
    with _LOCK:
        return _build(prefix, docs, extra)


def _build(prefix: str, docs: list[Document], extra: str = "") -> Chroma:
    """Open the collection for this fingerprint, building it once. Older collections with the same prefix are removed."""
    DIR.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{_fingerprint(extra)}"
    st = Chroma(collection_name=name, embedding_function=embedder()[0], persist_directory=str(DIR), collection_metadata={"hnsw:space": "cosine"})
    for old in [c.name for c in st._client.list_collections() if c.name.startswith(prefix + "_") and c.name != name]:
        st._client.delete_collection(old)
    if st._collection.count() == 0:
        st.add_documents(docs, ids=[d.metadata["id"] for d in docs])
    return st


@lru_cache(maxsize=1)
def clause_store() -> Chroma:
    return _store("clauses", [
        Document(page_content=clause_text(c), metadata={
            "id": c.id,
            "passage_id": c.id,
            "clause": c.clause_number,
            "title": c.title,
            "rule_type": c.rule_type,
            "source_document": "arogya/data/policy_clauses.json",
            "source_date": "unknown",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
        for c in CLAUSES
    ], extra="provenance-v1")


@lru_cache(maxsize=1)
def term_store() -> Chroma:
    return _store("terms", [Document(page_content=f"{k} {v['normalized']}", metadata={"id": f"T{i}", "term": k, **{kk: str(vv) for kk, vv in v.items()}}) for i, (k, v) in enumerate(TERM_MAP.items())])


def _docs(st: Chroma, query: str, k: int) -> list[Document]:
    out = []
    for d, dist in st.similarity_search_with_score(query, k=k):
        d.metadata = {**d.metadata, "score": round(1 - float(dist), 3), "mode": "vector"}      # cosine similarity
        out.append(d)
    return out


def semantic_clauses(query: str, k: int = 4, min_sim: float = 0.15) -> list[Document]:
    return [d for d in _docs(clause_store(), query, k) if d.metadata["score"] >= min_sim]


def rewrite_query(query: str) -> str:
    """Append a semantic glossary match and exact known claim-language expansions."""
    expansions = []
    query_vector = np.asarray(embedder()[0].embed_query(query))
    glossary_vectors = _glossary_embeddings()
    similarities = glossary_vectors @ query_vector / (
        np.linalg.norm(glossary_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-9
    )
    best_index = int(np.argmax(similarities))
    if similarities[best_index] >= 0.35:
        glossary = GLOSSARY[best_index]
        expansions.append(f"{glossary['topic']} {glossary['text']}")

    normalized_query = query.lower()
    for alias, details in sorted(TERM_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_query):
            expansions.append(f"{details['normalized']} {details['primaryClause']}")
    return " ".join([query, *dict.fromkeys(expansions)])


def rerank_clauses(query: str, candidates: list[Document], top: int = 4) -> list[Document]:
    """Rerank fused candidates using query-term coverage and exact phrase matches."""
    if not candidates or top <= 0:
        return []
    terms = set(tokenize(query))
    if not terms:
        return candidates[:top]

    max_fusion = max((float(d.metadata.get("score", 0)) for d in candidates), default=0) or 1
    ranked = []
    for document in candidates:
        content = f"{document.metadata.get('title', '')} {document.page_content}".lower()
        content_terms = set(tokenize(content))
        coverage = len(terms & content_terms) / len(terms)
        phrase_match = 1.0 if len(terms) > 1 and query.lower().strip() in content else 0.0
        fusion = float(document.metadata.get("score", 0)) / max_fusion
        score = 0.20 * coverage + 0.75 * fusion + 0.05 * phrase_match
        ranked.append((score, document))

    ranked.sort(key=lambda result: result[0], reverse=True)
    return [
        Document(page_content=document.page_content, metadata={
            **document.metadata,
            "rerank_score": round(score, 4),
            "mode": f"{document.metadata.get('mode', 'hybrid')}+rerank",
        })
        for score, document in ranked[:top]
    ]


def semantic_term(term: str, min_sim: float = 0.45) -> dict | None:
    """Nearest standard bill head for a hospital's own wording (used when no exact or substring match exists)."""
    hit = next(iter(_docs(term_store(), term, 1)), None)
    if not hit or hit.metadata["score"] < min_sim:
        return None
    return TERM_MAP[hit.metadata["term"]] | {"matchedTerm": hit.metadata["term"], "similarity": hit.metadata["score"]}


def similar_claims(text: str, k: int = 5, exclude: str = "") -> list[dict]:
    from ..data.synthetic_claims import load_claims
    docs = [Document(page_content=f"{c.diagnosis}. {c.procedure_name}. {c.icd10_code}. {c.hospital_name}", metadata={"id": c.claim_id, "claim": c.claim_id, "patient": c.insured_name,
                     "claimed": float(c.claimed_total)}) for c in load_claims()]
    st = _claims_store(tuple(d.metadata["id"] for d in docs), docs)
    return [{**d.metadata, "text": d.page_content} for d in _docs(st, text, k + 1) if d.metadata["claim"] != exclude][:k]


_CS: dict = {}


def _claims_store(key, docs):
    if key not in _CS:
        _CS[key] = _store("claims", docs, extra="|".join(key))
    return _CS[key]


def rrf(*ranked: list[Document], k: int = 60, top: int = 4, weights: tuple = (1.0, 2.0)) -> list[Document]:
    """Reciprocal rank fusion: a clause ranked well by either list rises, and by both rises most. Vectors get double weight, chosen on the development questions only."""
    score: dict[str, float] = {}
    keep: dict[str, Document] = {}
    modes: dict[str, set] = {}
    for w, lst in zip(weights, ranked):
        for i, d in enumerate(lst):
            key = d.metadata["clause"]
            score[key] = score.get(key, 0) + w / (k + i + 1)
            keep.setdefault(key, d)
            modes.setdefault(key, set()).add(d.metadata.get("mode", "bm25"))
    out = []
    for key in sorted(score, key=score.get, reverse=True)[:top]:
        d = keep[key]
        out.append(Document(page_content=d.page_content, metadata={**d.metadata, "score": round(score[key] * 61 / sum(weights), 3), "mode": "+".join(sorted(modes[key]))}))
    return out


def hybrid_candidates(query: str, k: int = 12) -> list[Document]:
    rewritten = rewrite_query(query)
    bm = [Document(page_content=d.page_content, metadata={**d.metadata, "mode": "bm25"})
          for d in ClauseRetriever(k=k).invoke(rewritten)]
    return rrf(bm, semantic_clauses(rewritten, k, min_sim=0.0), top=k)


class HybridClauseRetriever(BaseRetriever):
    """LangChain retriever: BM25 and the vector database, fused. Used by the adjudicator step of the workflow."""
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None) -> list[Document]:
        return rerank_clauses(query, hybrid_candidates(query, self.k * 3), top=self.k)


def hybrid_clauses(query: str, k: int = 4) -> list[Document]:
    return HybridClauseRetriever(k=k).invoke(query)


def hybrid_baseline_clauses(query: str, k: int = 4) -> list[Document]:
    """The original BM25 + vector RRF result, retained as an evaluation baseline."""
    bm = [Document(page_content=d.page_content, metadata={**d.metadata, "mode": "bm25"})
          for d in ClauseRetriever(k=k * 3).invoke(query)]
    return rrf(bm, semantic_clauses(query, k * 3, min_sim=0.0), top=k)


def bm25_clauses(query: str, k: int = 4) -> list[Document]:
    return [Document(page_content=d.page_content, metadata={**d.metadata, "mode": "bm25"}) for d in ClauseRetriever(k=k).invoke(query)]


def stats() -> dict:
    return {"embedder": embedder_name(), "clauses": clause_store()._collection.count(), "terms": term_store()._collection.count(), "store": "Chroma (cosine, persistent)", "path": str(DIR)}
