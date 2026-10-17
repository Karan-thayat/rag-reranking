# RAG Retrieval: Bi-Encoder Baseline vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked corpus of one complex PDF, compared on stress-test questions:

1. **Baseline:** `all-MiniLM-L6-v2` bi-encoder embeddings in a Chroma vector store → top 10 by cosine similarity → top 3 to the LLM (Answer 1).
2. **Re-ranking:** the same top 10 → `cross-encoder/ms-marco-MiniLM-L-6-v2` → re-sorted → top 3 to the LLM with the identical prompt (Answer 2).

**Document:** [NIST SP 800-63B-4](https://doi.org/10.6028/NIST.SP.800-63B-4), *Digital Identity Guidelines: Authentication and Authenticator Management* (July 2025, 129 pages). It is a technical policy published by the US government.

Everything is in one notebook, [`rag_reranking.ipynb`](rag_reranking.ipynb), saved with its outputs: PDF parsing, chunking, both pipelines, the failure-mode analysis (rank shifts plus Answer 1 vs. Answer 2), and latency profiling.

## Files

| Path | Contents |
|---|---|
| `rag_reranking.ipynb` | The notebook |
| `data/NIST.SP.800-63B-4.pdf` | The source document |
| `results/*.csv` | Tables written by the notebook: rank shifts, top-10 candidates, answers, latency, scaling |
| `requirements.txt` | Pinned dependencies (CPU-only PyTorch) |

## Running it

Tested with Python 3.12 on CPU. A full run takes about two minutes on a laptop.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...   # free key from https://console.groq.com
```

Then open `rag_reranking.ipynb` in Jupyter or VS Code with the `.venv` kernel and run all cells. Both models (about 90 MB each) download from the Hugging Face Hub on first use.

The LLM is `openai/gpt-oss-120b` on the Groq API. A run makes 14 calls, which fits within the free tier. Without `GROQ_API_KEY`, retrieval, re-ranking and latency profiling still run; only the answers are skipped. To use another model, change `LLM_MODEL` in the setup cell.
