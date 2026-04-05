# Chapter 27 — Choosing and Configuring Your RAG Pipeline

**Design of Agentic Systems with Case Studies**
INFO 7375: Prompt Engineering for Generative AI — Take-Home Midterm

---

## Architectural Claim

A RAG pipeline is a dependency chain of five lossy transformations: chunking, embedding, indexing, querying, and reranking. Information destroyed at any stage is invisible to every stage downstream. Upgrading the language model — the last stage — does not fix a failure at the first stage. **Architecture is the leverage point, not the model.**

---

## Repository Contents

| File | Description |
|------|-------------|
| `Chapter_27_Final_Integrated.md` | Publication-ready chapter with all Eddy revisions applied |
| `rag_pipeline_demo_chapter27.ipynb` | Runnable Jupyter notebook — two pipelines, one failure, zero API keys |
| `Authors_Note.md` | Human Decision Nodes, Eddy audit corrections, editorial process log |
| `eddy_revision_fix1.md` | Eddy the Editor — audit round 1 (Tetrahedron placement) |
| `eddy_revision_fix2.md` | Eddy the Editor — audit round 2 (legal case study, jargon fixes) |

---

## The Demo

The notebook builds two RAG pipelines over Python `requests` library documentation:

- **Pipeline A** — Fixed-size chunking (800 chars, no overlap) → *the failure case*
- **Pipeline B** — Structure-aware chunking (Markdown headers + code fences) → *the fix*

Everything else is identical: same corpus, same embedding model (`all-MiniLM-L6-v2`), same FAISS index, same query, same top-k. The only variable is the chunking architecture.

### What the notebook demonstrates

1. **AI Scaffold** — parses corpus, proposes pipeline config, halts for human review
2. **Mandatory Human Decision Node** — three AI proposals rejected and overridden with justification
3. **Deliberate failure triggered** — Pipeline A severs the retry logic section; the retriever returns prose without code; the LLM would hallucinate
4. **Controlled fix** — Pipeline B preserves the conceptual unit; the LLM receives complete context
5. **UMAP visualization** — shows why code chunks are geometrically unreachable in embedding space
6. **Exercises** — four guided experiments for the reader to break the system themselves

### Run it

```bash
pip install sentence-transformers faiss-cpu umap-learn matplotlib numpy scipy
jupyter notebook rag_pipeline_demo_chapter27.ipynb
```

No API keys required. Runs locally from a fresh clone.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| **Bookie the Bookmaker** | Drafted chapter prose (scenario → mechanism → design decision → failure → exercise) |
| **Eddy the Editor** | Two audit rounds — Tetrahedron placement, jargon-before-intuition, missing failure cases |
| **Figure Architect** | Five figure prompts for high-assertion zones (included in final chapter) |

---

## Chapter Structure (Tetrahedron: Structure–Logic–Implementation–Outcome)

| Element | Instantiation |
|---------|--------------|
| **Structure** | Five-stage pipeline: chunking → embedding → indexing → querying → reranking |
| **Logic** | Each stage is lossy; quality inequality chain bounds downstream performance |
| **Implementation** | Fixed-size vs. structure-aware chunking; general vs. domain embeddings; dense vs. hybrid retrieval |
| **Outcome** | Wrong chunking → hallucination that persists through model upgrades |

---

## Author

Mridula — Northeastern University, INFO 7375
