# Neural Search From Scratch

![python](https://img.shields.io/badge/python-3.8+-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![no-api](https://img.shields.io/badge/API_keys-none_required-brightgreen)

> Build a production-grade dense retrieval system from scratch — bi-encoder embeddings, HNSW graph indexing, and cross-encoder re-ranking. No Pinecone, no Weaviate, no API keys.

## What This Builds

- **Bi-Encoder** — encode queries and documents into dense vectors using local sentence-transformers
- **HNSW Index** — Hierarchical Navigable Small World graph for sub-linear approximate nearest neighbor search
- **Inverted Index** — optional sparse BM25 index for hybrid retrieval
- **Hybrid Fusion** — combine dense + sparse scores using Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Re-Ranker** — re-score top-k results with a more expensive cross-attention model
- **Persistence** — save/load index to disk with pickle + numpy

## Architecture

```
search/
├── encoder.py        # Bi-encoder using sentence-transformers (offline)
├── hnsw_index.py     # HNSW graph built from scratch
├── bm25.py           # BM25 sparse retrieval
├── hybrid.py         # Reciprocal Rank Fusion (dense + sparse)
├── reranker.py       # Cross-encoder re-ranking
└── search_engine.py  # Unified search API
```

## Quick Start

```bash
git clone https://github.com/SURENDER294/neural-search-from-scratch
cd neural-search-from-scratch
pip install -r requirements.txt

# Build index from a JSON document corpus
python run.py --index docs/corpus.json

# Search
python run.py --query "what is attention mechanism in transformers"
```

## Key Concepts

| Component | Algorithm | Complexity |
|---|---|---|
| Indexing | HNSW | O(n log n) |
| Dense search | Bi-encoder ANN | O(log n) avg |
| Sparse search | BM25 | O(k) |
| Re-ranking | Cross-encoder | O(k²) |

## Stack
- Python 3.8
- sentence-transformers (local models, no API)
- numpy, scipy
- No Pinecone, Weaviate, Qdrant, or any vector DB SaaS
