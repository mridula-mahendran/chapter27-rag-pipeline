# Chapter 27
## Choosing and Configuring Your RAG Pipeline — Chunking, Embedding, Indexing, Querying, and Reranking

---

### The Assistant That Invented a Method

In March 2024, a team of three engineers at a mid-stage startup built a retrieval-augmented generation assistant over the documentation for Python's `requests` library. The system ingested the library's full documentation — roughly forty pages of API reference, tutorials, and advanced usage guides — into a vector store. They chunked the documents at 512 tokens using a fixed-size sliding window, embedded them with OpenAI's `text-embedding-ada-002`, stored the vectors in Pinecone, and wired the retrieval output into GPT-3.5 Turbo with a system prompt instructing the model to answer questions using only the provided context.

For short, factual queries, the system worked. "What does `requests.get()` return?" produced a correct answer citing the `Response` object. "How do I set a timeout?" returned the correct keyword argument with an accurate code example. The team demoed the prototype to their engineering manager and began discussing production deployment.

Then a developer on another team typed: *"How do I implement retry logic with exponential backoff using the requests library?"*

The assistant returned a fluent, well-structured answer. It described a method called `.set_backoff()` on the `Session` object, explained its parameters — `max_retries`, `backoff_factor`, `status_forcelist` — and provided a code example that looked plausible to anyone who had used the library casually. The syntax was clean. The explanation was coherent. The method did not exist.

The `requests` library does not expose a `.set_backoff()` method. Retry logic in `requests` is implemented through the `urllib3.util.retry.Retry` class, mounted onto a `Session` via an `HTTPAdapter`. The correct implementation requires importing from `urllib3`, constructing a `Retry` object, wrapping it in an `HTTPAdapter`, and mounting that adapter onto a `Session`. The actual pattern spans three imports and roughly twelve lines of code, and in the documentation, the prose explanation and the code example sit in a section under the "Advanced Usage" header — a section that, in this pipeline, had been split across two chunks by the 512-token boundary.

The team's first instinct was to upgrade the model. They swapped GPT-3.5 Turbo for GPT-4. The hallucination did not disappear. It improved. The invented method was now described with greater nuance. GPT-4 added a deprecation warning for an older version of the fictitious API. It suggested checking the library's changelog for migration notes. The answer became more plausible, more detailed, and no more correct.

This is the chapter about why that happened, and what to do about it.

---

### The Question

Why does upgrading the language model not fix retrieval-augmented generation failures?

---

### The Pipeline You Already Built (and Did Not Design)

If you have built a RAG prototype, you have already made five architectural decisions, whether or not you were aware of making them. You chose how to chunk your documents. You chose an embedding model. You chose an index structure and a similarity metric. You chose a query strategy — how many chunks to retrieve, in what order, with what filtering. And you chose whether or not to rerank the retrieved results before passing them to the language model.

Most prototype builders make these decisions by default. They use whatever chunking the tutorial demonstrated, whatever embedding model the vector database's quickstart recommended, whatever top-k value seemed reasonable. The result is a pipeline that works on simple queries and fails on complex ones, and the builder has no vocabulary for diagnosing *where* in the pipeline the failure occurred.

This chapter gives you that vocabulary. More precisely, it gives you a repeatable diagnostic procedure.

**The diagnostic procedure is the Structure–Logic–Implementation–Outcome traversal.** Every time a stage is introduced in this chapter, it will be annotated with four questions in sequence:

- **Structure:** What is this stage's input/output contract — what does it take, what does it produce, what relationship to adjacent stages does that fix?
- **Logic:** Why does this stage necessarily lose information — what is the mathematical or distributional reason it cannot preserve everything it receives?
- **Implementation:** What are the specific configuration choices at this stage, and what information does each choice trade away?
- **Outcome:** What does a failure at this stage look like in the pipeline's observable behavior — what distinguishes a chunking failure from an embedding failure from an indexing failure?

By the time you finish this chapter, you will have practiced this traversal five times — once per stage. The goal is not that you know the vocabulary. The goal is that the traversal becomes automatic: when a pipeline returns a wrong answer, you move through the four questions at each stage in sequence, and you stop when you find the stage where the Logic explains the failure. You do not stop at the first stage where the Logic *sounds* plausible. You complete the traversal, because two stages can produce overlapping failure signatures, and the correct fix depends on which stage's Logic actually fits the evidence.

One preview of the procedure in action: an engineer on a customer support RAG system noticed that queries about refund policies returned correct answers, but queries containing specific order numbers — "What is the status of order ORD-29847?" — returned irrelevant chunks. The naive response would have been to swap the embedding model, increase top-k, or upgrade the LLM. Instead, the engineer ran the traversal at the querying stage. *Structure:* the querying stage takes a user question and retrieves by dense vector similarity. *Logic:* dense retrieval computes semantic similarity, which captures meaning but not exact string matches — an order number like "ORD-29847" carries no semantic content; it is a lexical identifier, and the embedding model was not broken, it was doing exactly what it was trained to do. *Implementation:* the pipeline used dense-only retrieval with no sparse or keyword component. *Outcome prediction:* adding BM25 sparse retrieval via hybrid search should recover exact-match queries without degrading semantic ones. The engineer added hybrid search with Reciprocal Rank Fusion, measured recall on both query types, and confirmed: semantic query recall stayed flat, exact-match recall jumped from 12% to 89% recall@5. The chunking was not the problem. The embedding model was not the problem. The fix was at exactly the stage the Logic identified, at the Implementation level, with a predicted Outcome confirmed by measurement.

This is the procedure. It is not the only way to debug a RAG pipeline. It is the way that does not waste time optimizing the wrong stage.

This chapter gives you a model of the RAG pipeline as a dependency chain of lossy transformations, where each stage's output constrains the ceiling of every stage downstream. The core claim is this:

**In a RAG pipeline, each stage is a lossy transformation. Chunking discards document structure. Embedding discards lexical detail. Indexing discards low-similarity candidates. Information destroyed at any stage is invisible to every stage downstream. The pipeline's failure mode is silent: it returns confidently wrong answers not because the model failed, but because the architecture filtered out the correct context before the model ever saw it. Upgrading the last stage — the language model — does not fix a failure at the first stage. The dependency runs forward. Diagnosis must run backward.**

If you understand this claim fully — not as a slogan, but as an operational principle you can use to diagnose and repair pipeline failures — you will never again respond to a hallucination by upgrading the model.

---

### Stage 1: Chunking — Where Structure Dies or Survives

> **Tetrahedron — Chunking**
>
> **Structure:** Chunking takes a document corpus and produces an ordered set of text segments. Each segment becomes an independent unit for all downstream stages. Nothing downstream can reconstruct the relationships between segments — the retriever will never see two chunks side by side and recognize that they belong together.
>
> **Logic:** Any chunking function that splits at boundaries not coinciding with the document's semantic boundaries necessarily severs some conceptual units. This is not a bug in any particular implementation — it is a consequence of the fact that fixed-size windows have no access to the document's meaning, only its token count. The information destroyed is not recoverable downstream: a severed code example is simply absent from every chunk the retriever will ever see.
>
> **Implementation:** The primary choice is the boundary criterion — token count (fixed-size), semantic signal (structure-aware), or sentence coherence (sentence-window). Each preserves different information and loses different information. Secondary choices are chunk size and overlap, each of which shifts the tradeoff between conceptual completeness and embedding quality.
>
> **Outcome:** A chunking failure produces answers that are fluent but structurally incomplete — the model has the concept but not the implementation, or the implementation but not the context that explains it. The diagnostic signal: the failure persists after you increase top-k, swap the embedding model, or add a reranker. None of those changes can retrieve a chunk that does not exist.

**What chunking actually does.** A chunking strategy defines a function *f* that maps a document *D* to an ordered set of text segments {*c*₁, *c*₂, ..., *cₙ*}. Each chunk *cᵢ* becomes an independent unit for all downstream stages: it will be embedded independently, indexed independently, retrieved independently, and — critically — it will be the unit of context the language model sees. Whatever information exists in the relationships *between* chunks is, for the retriever, gone.

This means chunking is not a preprocessing step. It is an information-theoretic decision about what the retriever is allowed to find.

**Fixed-size chunking** is the most common strategy in tutorials and the most dangerous in production. You specify a token count — 256, 512, 1024 — and the chunker walks through the document, cutting a new chunk every *n* tokens. Some implementations add overlap: a 512-token chunk with 64-token overlap means each chunk shares its last 64 tokens with the beginning of the next chunk.

The appeal is simplicity. The cost is that the chunk boundaries have no relationship to the document's semantic boundaries. A paragraph that explains a concept and then provides a code example may be split at the exact point where the prose ends and the code begins. A function signature and its docstring may land in different chunks. A section header — which tells the reader what the following content is about — may be separated from the content it introduces.

Return to the failure that opened this chapter. The `requests` library documentation discusses retry logic in a section under "Advanced Usage." The prose explanation — describing why you need retry logic, what exponential backoff means, and how the library's adapter pattern works — occupies roughly 380 tokens. The code example — showing the imports, the `Retry` object construction, the `HTTPAdapter` wrapping, and the `Session` mounting — occupies roughly 280 tokens. Together, they form a single conceptual unit of approximately 660 tokens.

A 512-token fixed-size chunker splits this unit at approximately the 512th token, which falls in the middle of the transition between the prose explanation and the code example. The first chunk contains the prose. The second chunk contains the code. When the user asks "How do I implement retry logic with exponential backoff?", the retriever computes similarity between the query embedding and all chunk embeddings. The prose chunk — which contains the words "retry," "exponential backoff," "requests," and "session" — has high cosine similarity to the query. The code chunk — which contains `from urllib3.util.retry import Retry`, `HTTPAdapter`, and `session.mount()` — has lower lexical overlap with the query and a somewhat different embedding profile, because code tokens and natural language tokens occupy different regions of most embedding spaces.

The retriever returns the prose chunk. The language model receives an explanation of the retry concept without the actual implementation. It is now in the position of a student who read the textbook paragraph but not the worked example, and has been asked to produce the solution. It does what any fluent generator does: it confabulates a plausible answer. The `.set_backoff()` method is invented not because GPT-3.5 is a poor model, but because the correct information — the code — was amputated from the context before the model ever saw it.

**Structure-aware chunking** replaces the fixed-size window with a parser that respects the document's own organizational signals. For Markdown documents, this means splitting at headers (`##`, `###`) and preserving code fences as atomic units. For HTML, it means splitting at `<section>`, `<article>`, or `<div>` boundaries. For Python docstrings, it means keeping each function's signature, docstring, and example together as a single chunk.

The implementation is straightforward. Libraries like LangChain's `MarkdownHeaderTextSplitter` or LlamaIndex's `SentenceSplitter` with metadata extraction provide off-the-shelf solutions. The key principle is that the chunker must be able to parse the document's structure, which means the document must *have* parseable structure. Markdown documentation, HTML pages with semantic markup, and code with docstrings are good candidates. Unstructured plain text — legacy PDFs, OCR output, raw transcripts — is not.

When the team in our opening example switched from fixed-size 512-token chunking to recursive structure-aware chunking that respected Markdown headers and kept code fences intact, the retry logic question was answered correctly. The prose and code were now in the same chunk. The retriever returned the complete conceptual unit. The language model generated the correct implementation. Nothing else in the pipeline changed — same embedding model, same index, same LLM.

This is the first lesson of pipeline architecture: **a failure that looks like a model problem is often a chunking problem, and a chunking problem is always an information destruction problem.**

**But structure-aware chunking is not free.** It produces variable-size chunks. Some sections of documentation are 200 tokens. Others are 2,000. This variability creates three downstream problems.

First, embedding models have input length limits and performance sweet spots. OpenAI's `text-embedding-ada-002` accepts up to 8,191 tokens, but its embedding quality degrades on very long inputs because the fixed-dimensional output vector (1,536 dimensions) must compress more information. Research from Muennighoff et al. (2023) on the MTEB benchmark shows that most embedding models perform best on inputs between 64 and 512 tokens, with diminishing returns beyond that range. A 2,000-token chunk may be semantically complete but poorly embedded.

Second, variable-size chunks create an uneven retrieval landscape. Cosine similarity between a short query embedding and a long chunk embedding behaves differently than between a short query and a short chunk. Longer chunks contain more semantic content, which can dilute the similarity signal for any single concept within the chunk. A 200-token chunk about retry logic and a 2,000-token chunk that mentions retry logic among fifteen other topics will produce different similarity scores for the same query, and the difference is not always in the direction you want.

Third, variable-size chunks complicate context window management. If you retrieve the top five chunks and each is 2,000 tokens, you have consumed 10,000 tokens of your context window before the model has generated a single token. If your context window is 8,192 tokens (GPT-3.5) or 128,000 tokens (GPT-4 Turbo), this may or may not matter — but you must account for it, and your retrieval strategy must be chunk-size-aware.

**The practical resolution** is not to pick the "best" chunking strategy in the abstract. It is to pick the strategy whose failure modes align least destructively with your specific documents and queries. If your documents are well-structured Markdown or HTML, structure-aware chunking will preserve conceptual units that fixed-size chunking would destroy. If your documents are unstructured plain text, you may need to use fixed-size chunking with generous overlap (128–256 tokens) and accept that some conceptual units will be split. If your chunks are highly variable in size, you may need a secondary pass that splits oversized chunks while preserving their metadata (the parent section header, for instance, prepended to each sub-chunk).

There is no universally correct chunking strategy. There are only strategies whose information loss patterns you understand and can compensate for downstream.

---

### Stage 2: Embedding — Where Meaning Becomes Geometry

> **Tetrahedron — Embedding**
>
> **Structure:** Embedding takes a text segment and produces a fixed-dimensional vector. All downstream retrieval operates on these vectors. The original text is no longer directly accessible to the retriever — only the geometry of its representation is.
>
> **Logic:** The fixed output dimension means embedding is a compression operation. More consequentially, the embedding model's training distribution determines what "similar" means in the resulting vector space. A model trained on natural language pairs will place code tokens and natural language tokens in different regions of the space — not because the model is failing, but because it learned a notion of similarity from data that treated them as different. This is the semantic gap: a query and its correct answer may be informationally equivalent but geometrically distant.
>
> **Implementation:** The primary choice is the embedding model, which defines the geometry of the search space. Secondary choices include whether to prepend metadata (section headers, document type) to chunks before embedding, whether to use separate models for different content types, and whether to fine-tune on domain-specific query-chunk pairs.
>
> **Outcome:** An embedding failure produces answers that are correct for one content type and systematically wrong for another — natural language queries retrieve well, but code lookups, legal citations, or numeric identifiers do not. The diagnostic signal is a categorical pattern in the failures, not a random one. Adding a reranker does not fix this: the correctly relevant chunks are not in the candidate set because the geometry placed them outside the retrieval window before the reranker ever ran.

**What embedding actually does.** An embedding model is a function *g* that maps a variable-length text string to a fixed-dimensional vector in ℝᵈ. For `text-embedding-ada-002`, *d* = 1,536. For `all-MiniLM-L6-v2` (a popular open-source model from Sentence Transformers), *d* = 384. The critical constraint is that *d* is fixed regardless of input length. A 50-token chunk and a 500-token chunk both produce vectors of the same dimensionality.

This means embedding is a compression operation, and compression always discards information. The question is *what* information is discarded and whether that information matters for your queries.

Modern embedding models are trained on contrastive objectives: given a query and a set of passages, the model learns to produce vectors where semantically similar pairs have high cosine similarity and dissimilar pairs have low cosine similarity. The training data typically consists of question-answer pairs, paraphrase pairs, and natural language inference pairs. This means the embedding model learns a notion of "similarity" that is shaped by its training distribution.

**The semantic gap.** Consider two chunks from the `requests` library documentation:

*Chunk A:* "To implement retry logic, you can use the `HTTPAdapter` class with a `Retry` configuration. This allows you to specify the number of retries, the backoff factor, and the HTTP status codes that should trigger a retry."

*Chunk B:*
```python
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

A human reading these chunks recognizes them as two representations of the same concept: one in prose, one in code. But embedding models trained primarily on natural language pairs may place these chunks in different regions of the embedding space. Chunk A contains the natural language tokens "retry logic," "backoff factor," and "HTTPAdapter," which align well with a natural language query. Chunk B contains code tokens — import statements, constructor calls, method invocations — that occupy a different distributional region.

The cosine similarity between the query "How do I implement retry logic with exponential backoff?" and Chunk A will typically be higher than the cosine similarity between that query and Chunk B, even though Chunk B is the more operationally useful result. This is the semantic gap: the embedding model's notion of similarity is trained on natural language, and code is not natural language.

**Seeing the gap, not just naming it.** Describing the semantic gap as "code and natural language occupy different regions of the embedding space" tells you what happened. It does not tell you why no parameter adjustment within dense retrieval can fix it. For that, you need to see the geometry.

Consider seven points in a two-dimensional projection of the embedding space, plotted with real chunk labels from the requests library case. Place the query — "How do I implement retry logic with exponential backoff?" — in the upper-left quadrant. Three prose chunks cluster near it: the retry logic explanation at cosine similarity 0.85, the timeout configuration overview at 0.82, and the error handling best practices at 0.79. These form a tight natural-language cluster. Two code chunks sit in a separate cluster in the lower-right quadrant: the retry code example containing `Retry()` and `HTTPAdapter` at cosine similarity 0.62, and the timeout code example with `try/except` at 0.58. One statutory chunk — the enumerated requirements of 26 U.S.C. § 501(c)(3) — sits far from both clusters in the lower-left at cosine similarity 0.31.

Now draw two circles centered on the query point. The first circle represents top-5 retrieval: its radius reaches to similarity 0.79, capturing the three prose chunks. The code chunks are outside it. The second circle represents top-20 retrieval: its radius extends further, and the code chunks are now barely inside it at positions 14–16. The statutory chunk is still outside both circles. Between the query and the code cluster, fill the space with small gray dots — the hundreds of other chunks ranked between the query and the correct answer, each one closer to the query in embedding space than the code chunk is.

This diagram makes three things simultaneously visible. First, why top-5 misses the code chunk — it is outside the retrieval radius, not ranked 6th but ranked 14th, with thirteen other chunks between it and the query. Second, why top-20 with a reranker might rescue the code chunk — it enters the candidate set at position 14, and the reranker can promote it once it's present. Third, why the statutory chunk is unreachable by any value of k in dense retrieval — it sits in a different region of the space entirely, and increasing the retrieval radius to include it means including everything else in between, which is most of the corpus.

The spatial framing makes a critical distinction concrete: the difference between *ranked low* and *geometrically unreachable*. A chunk ranked 6th can be promoted by increasing k to 10. A chunk in a different region of the space will not be promoted by any value of k, because there are hundreds of closer chunks between it and the query. You cannot tune your way across a structural gap in the space. The distance is not noise — it is the embedding model's training distribution made visible. This is what motivates the architectural fix — hybrid search, model swap, chunk preprocessing — rather than the parametric fix of adjusting k or the similarity threshold. Without the geometry, students keep tuning. With it, they can see why tuning cannot work and must start thinking about structural changes.

**What the diagram does not show.** Two-dimensional projections of 1,536-dimensional spaces preserve local neighborhood structure — which points are near which other points — but distort global distances. Two points that appear far apart in the projection may be moderately close in the original space, and two points that appear close may be close only because the projection algorithm collapsed a complex high-dimensional structure into a flat plane.

A student who reads the diagram too literally will make two specific mistakes. First, they will treat the clean separation between the prose cluster and the code cluster as a real boundary — concluding that content types are neatly separable in embedding space. In 1,536 dimensions, the clusters overlap in ways the projection hides. Some code-heavy prose chunks and some well-commented code chunks sit in the transition zone between regions. The student who believes the clean separation will be confused when the embedding model retrieves a code chunk for a prose query, treating it as an error when it is actually a chunk that genuinely straddles both distributional regions. Second, they will treat absolute distances in the plot as meaningful — concluding that the statute is "twice as far" as the code chunk because it looks that way in the projection. In the actual 1,536-dimensional space, the ratio may be entirely different.

Use the diagram to understand neighborhood structure: which chunks are in retrievable proximity to a query, and which are not. Do not use it to estimate actual similarity scores, to draw hard decision boundaries between content types, or to predict exact rankings. When the diagram suggests a chunk is in a separate neighborhood, verify the hypothesis by computing cosine similarity in the original space before committing to an architectural fix. The diagram is a map, not the territory, and like all two-dimensional maps of higher-dimensional spaces, it lies about distances while telling the truth about neighborhoods.

**In your own pipeline**, this diagram is reproducible. Embed a representative sample of your chunks — 200 to 500 is sufficient — along with a representative query. Run UMAP or t-SNE to project to two dimensions. Plot the query point and draw circles at your current top-k and top-20 retrieval radii. Identify which content types fall inside each circle and which fall outside. For each content type outside the top-k radius, compute the actual cosine similarity in the original space to determine whether you are looking at a chunk that is ranked low but reachable, or a chunk that is in a fundamentally different region. The projection tells you where to look. The original-space similarity score tells you what you found.

The requests library case demonstrates one version of the semantic gap: code tokens and natural language tokens occupy different regions of the embedding space because they have different lexical profiles. But this framing can mislead students into believing the semantic gap is a *content-type* problem — that it only appears at the boundary between code and prose. A second failure case breaks that generalization.

A paralegal at a mid-sized litigation firm queried their RAG assistant: *"What are the eligibility requirements under 26 U.S.C. § 501(c)(3) for tax-exempt status?"* The corpus contained both internal legal briefs authored by the firm's attorneys and the actual text of the U.S. Code. The system returned three chunks — all from internal briefs. The briefs discussed Section 501(c)(3), referenced it, argued about its interpretation. The actual statutory text — the chunk containing the enumerated eligibility requirements straight from the U.S. Code — ranked 19th by embedding similarity, outside the top-5 retrieval window.

The paralegal received a fluent summary of the firm's *interpretation* of the statute rather than the statute itself. She included this summary in a draft memorandum as if it were a direct recitation of the law. A senior attorney caught the error during review: the memo attributed requirements to the statute that were actually the firm's own analytical conclusions from a prior case, not the statutory language. The filing was corrected before submission, but the near-miss cost eight hours of senior attorney review time and delayed the filing by two days.

The firm's first response was to add a cross-encoder reranker. It did not help. The statute chunk was not in the top-20 candidate set — the embedding model had placed statutory text so far from the query embedding that it was geometrically unreachable by the retriever. The reranker can reorder the candidate set. It cannot retrieve a chunk that was never in it. This is the inequality in operation: *quality(R′) ≤ quality(R)*. The reranker's ceiling was set by the retriever's recall, and the retriever's recall was set by the embedding model's geometry.

The fix that worked was hybrid search with BM25 sparse retrieval. The exact string "26 U.S.C. § 501(c)(3)" appeared in both the query and the statute chunk. Sparse retrieval ranked the statute chunk first because lexical overlap is insensitive to distributional distance — it does not care whether the embedding model was trained on statutory text. Dense retrieval alone never would have found it.

**The mechanism is not content type. It is training distribution.** The brief and the statute are both English text. But they have different lexical profiles: the brief uses argumentative, discursive language; the statute uses enumerative, definitional language. A general-purpose embedding model trained on web text, Wikipedia, and question-answer pairs has seen vastly more argumentative prose than statutory enumeration. It embeds the brief accurately — that register is close to its training distribution — and embeds the statute poorly — that register is far from it. The query, phrased in conversational English, has high cosine similarity to the brief chunk and low cosine similarity to the statute chunk. The user gets a summary of someone's argument about the law instead of the law itself.

The student who diagnoses this correctly as a training-distribution mismatch will often propose the intuitive fix: swap the general-purpose embedding model for a legal-domain model fine-tuned on case law, briefs, and statutes. This improves statutory retrieval. But the same firm's corpus also contains plain-English client intake forms, email correspondence, and meeting notes. The legal-domain model was not trained on casual business English. Now those chunks become the distributionally distant ones. The client who wrote "we got our 501c3 status last year, do we still qualify?" uses colloquial language the legal model handles worse than the general-purpose model did.

Embedding model swaps do not eliminate the semantic gap. They move it. Every embedding model has a training distribution, and every training distribution has a boundary. Content outside that boundary is poorly represented regardless of how well content inside it is handled. When the corpus is heterogeneous — when it contains multiple registers, styles, or content types with different lexical profiles — a single embedding model will always have blind spots. The practical responses include hybrid search (which bypasses the embedding space for exact matches), prepending natural language descriptions to non-standard chunks before embedding, or maintaining separate indexes with different models and merging results. Which response is correct depends on which content type matters most for which query type. That is a product decision, not an engineering one, and it belongs to the human in the loop — not to the embedding model, the retriever, or the reranker.

The requests library case and the legal document case together establish a generalizable principle: **the embedding model's failure mode is always a distributional distance problem, never a content-type problem per se.** Code fails because code is distributionally distant from natural language training data. Statutes fail because statutory enumeration is distributionally distant from discursive prose training data. Clinical shorthand, mathematical notation, chemical formulas, and log files fail for the same reason: their lexical and syntactic profiles are far from the centroid of whatever data the model was trained on. The content type is a proxy. The training distribution is the cause.

**Choosing an embedding model** is therefore not a matter of picking the highest-scoring model on a general benchmark. It is a matter of matching the embedding model's training distribution to your document and query types. If your corpus is predominantly code, a code-trained embedding model (such as OpenAI's `code-search-ada-code-001` or Voyage AI's `voyage-code-3`) will produce a vector space where code chunks and code-related queries are closer together. If your corpus mixes prose and code, you may need a model trained on both modalities, or you may need to embed code chunks with a supplementary natural language description prepended to the chunk.

This is not a minor tuning decision. Switching embedding models can change retrieval recall by 10–30 percentage points on domain-specific benchmarks, according to evaluations on the MTEB leaderboard. The embedding model defines the geometry of your search space, and geometry determines what "near" means. If "near" does not align with "relevant," no amount of downstream sophistication will compensate.

**Dimensionality and the curse of information.** A 1,536-dimensional vector sounds like it should have enough capacity to represent any chunk's meaning. But consider what that vector must encode. It must capture the chunk's topic, its specificity, its tone, its relationships to other concepts, its key entities, and enough of its lexical content to distinguish it from similar chunks. All of this must fit into 1,536 floating-point numbers. For short, focused chunks (64–256 tokens), this compression is usually adequate. For long, multi-topic chunks (1,000+ tokens), the vector must represent multiple concepts simultaneously, and the result is a centroid — a point in embedding space that is close to the average of the chunk's constituent meanings and therefore not particularly close to any single one of them.

This is why chunk size and embedding quality are coupled. The chunking stage determines how much semantic content each embedding must compress. If your chunks are too large, your embeddings become blurry — they point vaguely toward the right region of the space but do not resolve the specific concept the query needs. If your chunks are too small, your embeddings become sharp but fragmented — each one captures a single idea precisely, but the relationships between ideas are lost.

The practical heuristic supported by current benchmarks: for most general-purpose embedding models, the sweet spot is 128–512 tokens per chunk. Below 128, you lose inter-sentence coherence. Above 512, you lose embedding specificity. These bounds shift with embedding model architecture and training — always validate on your specific corpus and query distribution.

---

### Stage 3: Indexing — Where Geometry Meets Approximation

> **Tetrahedron — Indexing**
>
> **Structure:** Indexing takes the full set of chunk vectors and organizes them into a data structure that supports approximate nearest-neighbor search. For any given query, the index returns a candidate set — not the full corpus. Everything outside the candidate set is invisible to every downstream stage.
>
> **Logic:** Approximate nearest-neighbor algorithms trade retrieval precision for speed by examining only a subset of vectors per query. The recall gap (typically 2–10% versus exact search) is not random — it disproportionately affects chunks whose vectors sit near the boundary between graph neighborhoods or Voronoi cells. These boundary cases are often the hard queries: the ones where the correct chunk is relevant but not the most semantically obvious match.
>
> **Implementation:** The primary choices are index type (HNSW, IVF, flat), search parameters (ef_search, nprobe), and metadata schema. Metadata filtering is the index stage's mechanism for reintroducing document structure that chunking destroyed — filtering by source document or section type before computing vector similarity removes irrelevant candidates that would otherwise compete for top-k slots.
>
> **Outcome:** An indexing failure is the hardest to distinguish from an embedding failure, because both manifest as relevant chunks not appearing in retrieval results. The diagnostic signal specific to indexing is intermittency: the failure affects some queries but not others with similar embeddings. The test — increase ef_search or nprobe and re-run without changing anything else. If recall improves, the failure was at the index. If it does not, the failure is upstream.

**Exact nearest neighbor search** computes the cosine similarity (or Euclidean distance, or dot product) between the query vector and every vector in the index. For a corpus of *N* chunks, this requires *N* distance computations per query. When *N* is 10,000, this is fast. When *N* is 10 million, this takes seconds per query — unacceptable for a real-time assistant.

**Approximate nearest neighbor (ANN) search** trades retrieval precision for speed. Algorithms like HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), and PQ (Product Quantization) organize the vector space into navigable structures that allow the search to examine only a subset of vectors for each query.

HNSW builds a multi-layer graph where each vector is a node, and edges connect nodes that are close in the embedding space. A query traverses the graph from a random entry point, greedily following edges toward the query vector. The parameter `ef_search` controls how many candidates the algorithm examines — higher values produce more accurate results but slower queries. At `ef_search = 64`, a typical HNSW index over 1 million vectors achieves ~98% recall compared to exact search. At `ef_search = 16`, recall drops to ~90%.

That 2–10% recall gap means that for every hundred queries, between two and ten will fail to retrieve a chunk that exact search would have found. These are not random failures — they disproportionately affect queries where the correct chunk's embedding is on the boundary between graph neighborhoods. In practice, these are often the hard queries: the ones where the answer requires a chunk that is relevant but not the most obvious match.

**IVF** partitions the vector space into *k* Voronoi cells using k-means clustering, then searches only the `nprobe` nearest cells to the query. If the correct chunk's vector sits near the boundary between two cells and the query's nearest cell is the other one, the chunk is missed. Increasing `nprobe` from 1 to 10 typically raises recall from ~60% to ~95%, but at the cost of 10× more distance computations.

**Product Quantization** compresses vectors from 32-bit floats to 8-bit or 4-bit codes, reducing memory by 4–8× at the cost of introducing quantization error into distance calculations. A chunk that was marginally the nearest neighbor in the full-precision space may no longer be the nearest neighbor in the quantized space.

The point is not that ANN indexes are bad. They are necessary for any corpus above a trivial size. The point is that indexing is a lossy transformation with quantifiable recall loss, and that recall loss compounds with the losses already introduced by chunking and embedding. If chunking destroyed a conceptual unit and embedding blurred a code chunk's representation, the ANN index's 2% recall gap may be the third and final filter that ensures the correct context never reaches the language model.

**Metadata filtering** is the indexing stage's most underused tool. Most vector databases support attaching metadata to each chunk — the source document, the section header, the document type, the date, custom tags — and filtering the search space before computing vector similarity. If the user's query is about the `requests` library and your index contains documentation for fifty Python libraries, a metadata filter that restricts the search to `source = "requests"` eliminates 98% of the candidate space before ANN search begins. This is not an optimization. It is a precision instrument that removes irrelevant candidates that would otherwise compete with the correct chunk for the top-k slots.

You should think of metadata filtering as the indexing stage's mechanism for reintroducing document structure that the chunking stage destroyed. The section header that was severed from its content during chunking can be stored as metadata and used as a filter during retrieval. This is inelegant — it is a patch for an earlier stage's information loss — but it works, and in production systems it is often the difference between a pipeline that retrieves well and one that does not.

---

### Stage 4: Querying — Where the User Meets the Pipeline

> **Tetrahedron — Querying**
>
> **Structure:** The querying stage takes a user question and the index and produces a ranked candidate set. It is the only stage where the user's intent directly shapes retrieval — all prior stages operate on the corpus without access to what will eventually be asked.
>
> **Logic:** The user's raw question is often not the optimal retrieval query. Dense retrieval computes a single similarity score between the query embedding and each chunk embedding. It cannot retrieve two separately indexed concepts and recognize that both are needed to answer one question. For complex multi-concept queries, this is a structural limitation of the retrieval operation, not a failure of any component.
>
> **Implementation:** The choices are query transformation strategy (raw, HyDE, multi-query decomposition), retrieval modality (dense, sparse, hybrid), top-k value, and filtering criteria. Each choice determines the shape and content of the candidate set passed downstream.
>
> **Outcome:** A querying failure on complex multi-concept questions is the most common misdiagnosis in pipeline debugging, because its surface appearance resembles a chunking failure. Consider: "How do I configure retry logic with custom backoff AND connection pooling in the same session?" A student running the traversal may anchor at the chunking stage — the retry logic and connection pooling sections are in separate chunks, so enlarging chunks should fix it. Increasing chunk size to 2,000 tokens does not improve recall. The concepts were intact in their individual chunks; the problem was that a single retrieval pass returns one cluster of semantically similar content, not two independent concepts. The correct fix is multi-query decomposition: break the query into sub-queries, retrieve independently, merge results. This is the traversal's own failure mode: stop at the first stage where the Logic sounds plausible, and you will implement the wrong fix. Complete the traversal. Check whether a downstream stage fits the evidence better before committing to a fix.

**The naive query strategy** embeds the user's question, computes similarity against the index, and returns the top *k* chunks ranked by similarity score. The parameter *k* — typically set to 3, 5, or 10 — determines how many chunks the language model sees. This strategy is simple, and its failure modes are predictable.

If *k* is too small, the correct chunk may be ranked 6th and never seen. If *k* is too large, the context window fills with marginally relevant chunks that dilute the model's attention. Research by Liu et al. (2023) on the "lost in the middle" phenomenon demonstrated that language models attend most strongly to information at the beginning and end of the context window, with significant attention degradation for content in the middle positions. Stuffing fifteen chunks into the context is not fifteen times better than stuffing one — it may be worse, if the correct chunk is buried at position eight.

**Query transformation** is the first lever for improving retrieval without touching the index. The user's raw query is often not the best retrieval query. "How do I implement retry logic with exponential backoff using the requests library?" is a natural question but a suboptimal retrieval query — it contains conversational filler ("How do I"), a broad action verb ("implement"), and a specific technical concept ("retry logic with exponential backoff") mixed with a library name ("requests"). The embedding of this query is a compromise between all of these semantic signals.

**HyDE (Hypothetical Document Embeddings)**, introduced by Gao et al. (2022), addresses this by asking the language model to generate a hypothetical answer to the question *before* retrieval. The hypothetical answer is then embedded, and the embedding is used as the retrieval query. The intuition is that the hypothetical answer's embedding will be closer to the actual answer's embedding than the question's embedding is, because answers and documents occupy similar regions of the embedding space, while questions and documents do not.

In practice, HyDE helps most when the query is abstract or conceptual ("What is the design philosophy behind the adapter pattern in requests?") and helps least when the query is specific and lexically aligned with the target document ("What are the parameters of urllib3.util.retry.Retry?"). It also introduces latency — one additional LLM call per query — and can propagate errors if the hypothetical answer is wrong in a way that biases the retrieval toward the wrong region of the embedding space.

**Multi-query retrieval** generates multiple reformulations of the user's question and retrieves for each one independently, then merges and deduplicates the results. The reformulations can be generated by the language model ("Rephrase this question in three different ways") or by systematic decomposition ("Break this question into sub-questions"). The benefit is diversity — different phrasings activate different regions of the embedding space, increasing the probability that at least one retrieval query finds the correct chunk. The cost is latency (multiple embedding and retrieval operations) and complexity (merging ranked lists).

**Hybrid search** combines dense vector similarity with sparse keyword matching (BM25 or TF-IDF). This is one of the most consistently effective query strategies in the current literature, because dense and sparse retrieval have complementary failure modes. Dense retrieval excels at semantic matching ("implement retry logic" → chunks about retrying failed requests) but misses exact keyword matches. Sparse retrieval excels at lexical matching ("urllib3.util.retry.Retry" → chunks containing that exact string) but misses semantic paraphrases. Combining both — typically with a weighted sum of normalized scores, known as Reciprocal Rank Fusion (RRF) — captures chunks that either method alone would miss.

The weight between dense and sparse scores is a tunable parameter. A 70/30 split favoring dense retrieval is a common starting point for natural language queries over documentation, but the optimal ratio depends on your query distribution. If your users frequently search for specific class names, function signatures, or error messages, increase the sparse weight. If your users ask conceptual questions in everyday language, increase the dense weight. Measure retrieval recall on a representative query set, not on intuition.

---

### Stage 5: Reranking — The Last Chance to Correct the Pipeline

> **Tetrahedron — Reranking**
>
> **Structure:** Reranking takes the candidate set from querying and produces a reordered, typically smaller subset for the language model. It operates only on what it receives. It has no access to the full corpus.
>
> **Logic:** Cross-encoder reranking is strictly more accurate than bi-encoder similarity for relevance scoring — it attends jointly to the query and each candidate, capturing fine-grained interactions that vector similarity cannot. But "more accurate at ordering" is not the same as "more accurate at recall." The reranker's ceiling is the recall of the querying stage. It can reorder the candidates. It cannot retrieve candidates that were never returned.
>
> **Implementation:** The choices are reranker model, candidate set size (typically 20–50 for reranking down to 3–5), and whether to apply a relevance threshold to filter out low-scoring candidates.
>
> **Outcome:** Reranking failures are almost always misattributed upstream failures — the reranker is blamed for returning wrong answers when the correct chunk was never in the candidate set. The diagnostic test: inspect the full candidate set before reranking. If the correct chunk is present at position 14 and the reranker promotes it to position 2, the reranker is working correctly. If the correct chunk is absent from the candidate set entirely, the failure is upstream — at chunking, embedding, indexing, or querying — and adding or improving the reranker will not help.

**Why reranking exists.** The retrieval stage returns the top *k* chunks ranked by embedding similarity. But embedding similarity is computed independently for each chunk — the retriever does not consider how the chunks relate to each other, whether they are redundant, or whether a chunk that is individually less similar to the query is more useful in the context of the other retrieved chunks. Embedding similarity is also a coarse signal: two chunks with cosine similarity 0.82 and 0.80 are treated as meaningfully different, but the difference may be noise.

A **cross-encoder reranker** takes a (query, chunk) pair as input and produces a relevance score using a model that attends jointly to the query and the chunk. Unlike the bi-encoder used for embedding (which encodes query and chunk independently and compares their vectors), the cross-encoder processes the concatenated query-chunk pair through a full transformer, allowing it to capture fine-grained interactions between query tokens and chunk tokens. This is dramatically more expensive — O(*k*) forward passes through a large model instead of a single vector comparison — but dramatically more accurate.

On the BEIR benchmark, adding a cross-encoder reranker (such as `cross-encoder/ms-marco-MiniLM-L-12-v2` from Sentence Transformers, or Cohere's `rerank-english-v3.0`) to a bi-encoder retrieval pipeline improves nDCG@10 by 5–15 percentage points depending on the dataset. In practical terms, this means a chunk that was ranked 7th by embedding similarity — outside a top-5 retrieval window — may be promoted to rank 2 by the reranker, bringing it into the context the language model sees.

**The reranking workflow** is: retrieve a larger candidate set (top 20–50 by embedding similarity), score each candidate with the cross-encoder, sort by cross-encoder score, and pass the top *k* (typically 3–5) to the language model. The initial retrieval is broad and fast; the reranking is narrow and precise. This two-stage architecture — recall-optimized retrieval followed by precision-optimized reranking — is the standard pattern in information retrieval, and there is no good reason to skip it in a production RAG pipeline.

**What reranking cannot fix.** If the correct chunk was not in the initial candidate set — because it was destroyed by chunking, poorly embedded, or missed by the ANN index — the reranker cannot promote it. Reranking operates on the candidate set, not on the full corpus. It is a precision tool, not a recall tool. It can reorder the candidates. It cannot recall candidates that were never retrieved.

This is the fundamental constraint of the pipeline model: **each stage can only degrade or maintain the information available to it. No stage can recover information that a prior stage destroyed.** The reranker cannot fix a chunking error. The embedding model cannot fix a metadata problem. The language model cannot fix a retrieval failure. The pipeline is a dependency chain, and information flows in one direction.

---

### The Dependency Chain: A Formal View

It is worth stating the pipeline's information flow precisely, because precision prevents the debugging error of fixing the wrong stage.

Let *D* be the original document corpus. Define:

- **Chunking**: *C* = chunk(*D*) — transforms documents into chunks, discarding inter-chunk relationships and (in fixed-size chunking) intra-concept coherence.
- **Embedding**: *E* = embed(*C*) — transforms chunks into fixed-dimensional vectors, discarding information that exceeds the vector's representational capacity.
- **Indexing**: *I* = index(*E*) — organizes vectors for retrieval, potentially discarding candidates through ANN approximation.
- **Querying**: *R* = query(*I*, *q*) — retrieves top-k candidates given a user query *q*, constrained by the query strategy and the index structure.
- **Reranking**: *R′* = rerank(*R*, *q*) — reorders (and possibly filters) the retrieved candidates based on fine-grained relevance scoring.
- **Generation**: *a* = generate(*R′*, *q*) — the language model produces an answer given the reranked context and the query.

The answer quality is bounded by a chain of inequalities:

*quality(a) ≤ quality(R′) ≤ quality(R) ≤ quality(I) ≤ quality(E) ≤ quality(C) ≤ quality(D)*

This is an inequality, not an equality. Each stage can lose information. No stage can create it. If the relevant content was split across two chunks at the chunking stage (*quality(C) < quality(D)*), then even a perfect embedding model, a perfect index, a perfect query strategy, a perfect reranker, and a perfect language model will produce an answer that is missing that content. The pipeline's ceiling is set by its weakest stage, and the weakest stage is usually the first one.

This is why upgrading the model — the last stage — does not fix a retrieval failure. GPT-4 is a better generator than GPT-3.5. But a better generator operating on the same incomplete context will produce a better-written wrong answer, not a correct one. The hallucination improves in fluency because the model improves. It does not improve in correctness because the context does not improve. The failure is upstream.

---

### The Forensic Case: Diagnosing a Pipeline Stage by Stage

Let us now reconstruct the opening failure in full diagnostic detail, treating the pipeline as a system to be instrumented rather than a black box to be upgraded.

**The query:** "How do I implement retry logic with exponential backoff using the requests library?"

**Stage 1 — Chunking.** The documentation section on retry logic occupies approximately 660 tokens. The fixed-size chunker at 512 tokens with no overlap splits this section into two chunks:
- Chunk A (tokens 1–512): The prose explanation of retry logic, including the concepts of exponential backoff, the `HTTPAdapter` pattern, and the `Retry` class.
- Chunk B (tokens 513–660): The code example showing the imports, object construction, and session mounting.

The section header ("Advanced Usage: Retry Logic") is in Chunk A. The code is in Chunk B. The conceptual unit has been severed. This is the point of information destruction.

**Stage 2 — Embedding.** Both chunks are embedded. Chunk A's embedding captures natural language semantics: "retry," "exponential backoff," "requests," "HTTPAdapter." Chunk B's embedding captures code tokens: `from urllib3.util.retry import Retry`, `HTTPAdapter(max_retries=retries)`, `session.mount()`. The two embeddings are in different regions of the vector space because the embedding model was trained predominantly on natural language pairs.

**Stage 3 — Indexing.** Both chunks are indexed. No metadata filtering is applied. The index contains chunks from the entire `requests` documentation plus several other libraries.

**Stage 4 — Querying.** The query is embedded. Cosine similarity is computed. Chunk A ranks 2nd (high natural language overlap with the query). Chunk B ranks 14th (lower overlap due to code tokens). The retrieval strategy uses top-k = 5. Chunk B is outside the retrieval window.

**Stage 5 — Reranking.** No reranker is configured. The top 5 chunks are passed directly to the language model.

**Stage 6 — Generation.** The language model receives Chunk A (prose explanation without code) and four other marginally relevant chunks. It must generate a code example from a prose description. It does what language models do when they have partial information and a generation mandate: it confabulates. The `.set_backoff()` method is invented to fill the gap left by the missing code chunk.

**The fix.** Switching to structure-aware chunking produces a single chunk containing both the prose explanation and the code example (~660 tokens). This chunk is embedded as a unit. It ranks 1st for the query because it contains both the natural language description and the library-specific terms. The language model receives the complete conceptual unit. The generated answer is correct.

No other stage was changed. The embedding model, the index, the query strategy, the LLM — all identical. The fix was entirely at the chunking stage, because that is where the information was destroyed.

---

### Complication: Why There Is No Correct Default Configuration

The structure-aware chunking fix is satisfying, and it is specific to this failure mode. It does not generalize to a universal recommendation because every design choice in the pipeline involves tradeoffs that depend on the corpus, the query distribution, and the performance requirements.

**Chunking tradeoffs.** Structure-aware chunking requires parseable document structure. If your corpus is OCR output from scanned engineering reports, there are no Markdown headers to split on. If your corpus is a mix of well-structured API documentation and unstructured forum posts, the chunking strategy that works for one document type will fail for the other. A production pipeline over heterogeneous corpora often requires multiple chunking strategies with document-type routing — a complexity that no tutorial prepares you for.

Variable-size chunks also interact badly with some embedding models. If your chunks range from 50 to 2,000 tokens and your embedding model was trained on passages of 128–256 tokens, the short and long chunks may be represented with different levels of fidelity. The embedding space becomes inconsistent — similarity scores are not comparable across chunks of very different sizes, because the compression ratio is different.

**Embedding tradeoffs.** Larger embedding models (e.g., 3,072-dimensional vectors from `text-embedding-3-large`) capture more information per chunk but require more storage and slower similarity computation. For a corpus of 10 million chunks, the difference between 384-dimensional and 3,072-dimensional vectors is ~30 GB of additional storage and ~4× longer query times. Smaller models may be "good enough" for your query distribution, and the only way to know is to measure retrieval recall on a representative evaluation set.

Domain-specific embedding models (trained or fine-tuned on your document type) consistently outperform general-purpose models on domain-specific retrieval. Fine-tuning an embedding model on (query, relevant chunk) pairs from your domain typically improves recall@10 by 5–20% over the best general-purpose model. But fine-tuning requires labeled data — at least a few hundred query-chunk pairs — and ongoing maintenance as your corpus evolves.

**Indexing tradeoffs.** HNSW provides excellent recall at moderate corpus sizes (up to ~10 million vectors) but consumes significant memory because it stores the full graph in RAM. IVF with Product Quantization reduces memory by 4–8× but introduces quantization error. For a startup with 100,000 documentation chunks, HNSW with default parameters is fine. For a company with 50 million chunks across multiple languages, the index architecture becomes a systems engineering problem with real infrastructure cost.

**Querying tradeoffs.** Hybrid search (dense + sparse) is almost always better than dense-only, but it requires maintaining a separate keyword index (Elasticsearch, OpenSearch, or a vector database that supports both modalities). Multi-query retrieval improves recall but adds latency. HyDE improves semantic queries but can bias retrieval on specific queries.

**Reranking tradeoffs.** Cross-encoder reranking is the single highest-impact addition to most pipelines, but it adds 100–500ms of latency per query (depending on the reranker model and the candidate set size). For a real-time assistant, this may be acceptable. For a batch processing pipeline, it is negligible. For a system that must respond in under 200ms, it may require a faster reranker (a distilled model, or a lightweight reranker like ColBERT) or asynchronous architecture.

The point is not that these tradeoffs make the problem unsolvable. It is that they make the problem *specific*. There is no general-purpose RAG configuration that works well for all corpora, all query types, and all latency requirements. Every production pipeline is a set of tradeoffs, and the engineering skill is in making those tradeoffs deliberately rather than by default.

---

### Student Activities

**Exercise 27.1: The Retrieval Autopsy**

You are given a RAG pipeline with the following configuration:
- Corpus: The full `requests` library documentation (Markdown source)
- Chunking: Fixed-size, 512 tokens, no overlap
- Embedding: `text-embedding-ada-002`
- Index: HNSW with default parameters (Pinecone)
- Query: Top-5, dense only
- Reranker: None
- LLM: GPT-4 Turbo

The query is: "How do I implement retry logic with exponential backoff using the requests library?"

The system returns an answer that invents a `.set_backoff()` method.

(a) Instrument the pipeline. For each of the five stages, record the intermediate output: the chunks produced, the embeddings (at minimum, report the dimensionality and the cosine similarity between the query embedding and the top-10 chunk embeddings), the retrieval results (chunk text and similarity scores), and the final answer.

(b) Identify the stage at which the relevant information was lost. Provide the specific evidence from your instrumentation that supports your diagnosis.

(c) Propose a fix at the identified stage. Implement the fix. Re-run the query and record the new intermediate outputs and final answer.

(d) Without changing your fix from (c), add a cross-encoder reranker (use `cross-encoder/ms-marco-MiniLM-L-12-v2`). Retrieve the top 20 chunks, rerank, and pass the top 5 to the LLM. Does the reranker change the answer? Why or why not?

(e) Now revert to the original fixed-size chunking (512 tokens, no overlap), but add the reranker from (d). Re-run the query. Does the reranker fix the hallucination? At what position did the correct code chunk appear in the top-20 candidate set? What does this tell you about the relationship between recall and reranking?

**Exercise 27.2: Overlap as a Patch**

Starting from the original fixed-size 512-token chunking configuration:

(a) Add 64 tokens of overlap. Re-run the retry logic query. Is the code example now present in any retrieved chunk? Report the specific chunk text and its similarity ranking.

(b) Increase overlap to 128 tokens. Repeat. Then 256 tokens. For each overlap value, report retrieval recall for the retry logic query and two other queries of your choice (one simple factual query, one complex multi-concept query).

(c) Plot the relationship between overlap size and retrieval recall. At what point does increasing overlap stop improving recall for the complex query? Why?

(d) Calculate the storage overhead introduced by each overlap level (total tokens stored / original document tokens). Is the tradeoff between recall improvement and storage cost linear? Explain.

**Exercise 27.3: The Embedding Model Swap**

Using structure-aware chunking (so that the code example is intact in a single chunk):

(a) Embed all chunks with `text-embedding-ada-002`. Record the cosine similarity between the retry logic query and the code-containing chunk.

(b) Re-embed all chunks with a code-optimized embedding model (e.g., Voyage AI's `voyage-code-3` or `code-search-ada-code-001`). Record the cosine similarity for the same query-chunk pair.

(c) Design a query where the code-optimized model retrieves the correct chunk and the general-purpose model does not. Design a query where the reverse is true. What does this tell you about the relationship between embedding model training distribution and query-corpus alignment?

**Exercise 27.4: The Hybrid Search Experiment**

Configure a hybrid search pipeline that combines dense retrieval (embedding similarity) with sparse retrieval (BM25).

(a) For the retry logic query, report the top-5 results from dense-only, sparse-only, and hybrid (RRF with equal weights) retrieval. Which configuration ranks the correct chunk highest?

(b) Design a query that dense retrieval handles well but sparse retrieval handles poorly. Design the reverse. In each case, explain the mechanism: what does the successful retrieval method capture that the unsuccessful one misses?

(c) Vary the dense/sparse weight ratio from 100/0 to 0/100 in increments of 10. Plot retrieval recall@5 for three different query types: a natural language conceptual question, a specific function name lookup, and a hybrid question that mixes concepts and specific terms. At what weight ratio does each query type achieve maximum recall?

**Exercise 27.5: The End-to-End Pipeline Design Challenge (Open-Ended)**

You are tasked with building a RAG assistant over a new corpus: the complete documentation for a web framework of your choice (Django, FastAPI, Flask, Express, or Rails). The documentation includes API references, tutorials, how-to guides, and conceptual explanations. Your users are developers who ask questions ranging from simple API lookups ("What are the parameters of `Router`?") to complex implementation questions ("How do I implement OAuth2 with PKCE using this framework's middleware?").

Design and justify a complete pipeline configuration:
- Chunking strategy (with explicit justification for why your strategy's failure modes are acceptable for this corpus)
- Embedding model (with benchmarks or evaluation on at least 10 representative queries)
- Index configuration (type, parameters, metadata schema)
- Query strategy (dense, sparse, hybrid, with parameter values)
- Reranking (model choice, candidate set size, threshold)

For each stage, identify the failure mode you are *choosing to accept* and explain why it is less damaging than the alternatives. Your design document should make the tradeoffs explicit — not as a checklist, but as an argument for why this specific configuration is the best fit for this specific corpus and query distribution.

This exercise is not asking you to find the optimal configuration. It is asking you to demonstrate that you understand why a universal optimal configuration does not exist.

---

### Integrating LLMs as Diagnostic Scaffolding

The pipeline debugging process described in Exercise 27.1 is tedious. Instrumenting five stages, recording intermediate outputs, computing similarity scores, comparing chunk texts — this is high-effort, low-creativity work that is amenable to LLM assistance. But the assistance must be structured carefully to ensure the student retains the diagnostic reasoning.

**Appropriate LLM use:** Ask an LLM to write the instrumentation code — the logging, the similarity computation, the output formatting. This reduces the mechanical burden of pipeline instrumentation and frees the student to focus on interpreting the outputs. The LLM can generate a function that takes a query and returns a structured report of each stage's output. The student's job is to read the report and identify the failure stage.

**Appropriate LLM use:** After identifying a failure, ask an LLM to enumerate the possible fixes at the identified stage. The LLM can list chunking strategies, overlap values, embedding models, and reranking options. The student's job is to select the fix, justify the selection, and predict its effect before running the experiment.

**Inappropriate LLM use:** Asking an LLM to diagnose the failure directly ("Here is my pipeline output. What is wrong?"). This offloads the diagnostic reasoning to the model, and diagnostic reasoning is exactly the skill this chapter is designed to build. An LLM can tell you that the code chunk was not retrieved. It cannot tell you *why that matters for your specific pipeline*, because it does not know your performance requirements, your query distribution, or your acceptable failure modes.

**The human decision node:** At every point in the debugging process, the student must decide what "correct" means. Is the goal to retrieve the exact code example? Or is it sufficient to retrieve a chunk that describes the pattern conceptually? The answer depends on the user's intent, which depends on the application context, which depends on a product decision that no retrieval system can make. The student must make this judgment. The LLM scaffolds the execution. The student owns the diagnosis.

---

### Chapter Summary

A RAG pipeline is a dependency chain of five lossy transformations: chunking, embedding, indexing, querying, and reranking. Each stage's output constrains the ceiling of every stage downstream. Information destroyed at any stage is invisible to every subsequent stage, including the language model.

The pipeline's failure mode is silent. It does not throw an error when the correct context is chunked, embedded, indexed, or queried out of existence. It returns a fluent, confident answer generated from incomplete context. The hallucination is not a model failure. It is an architecture failure that the model makes invisible by generating plausible text.

Upgrading the language model does not fix retrieval failures. A better model operating on the same incomplete context produces a more plausible hallucination. The fix must be applied at the stage where the information was destroyed, and identifying that stage requires instrumenting the pipeline and inspecting intermediate outputs at each transformation.

There is no universally correct pipeline configuration. There are only configurations whose failure modes you understand and whose tradeoffs you have made deliberately. Fixed-size chunking fails when it severs conceptual units. Structure-aware chunking fails on unstructured text. General-purpose embeddings fail on code. Code embeddings fail on conceptual queries. Dense retrieval fails on keyword lookups. Sparse retrieval fails on semantic paraphrases. Every choice is a tradeoff, and the engineering skill is in making that tradeoff visible, measurable, and aligned with your specific documents and your specific users.

The pipeline is a system. Treat it as one.

---

### Figure Prompts

**Figure 27.1: The RAG Pipeline as a Dependency Chain of Lossy Transformations**
A horizontal flow diagram showing six stages left to right: Document Corpus → Chunking → Embedding → Indexing → Querying → Reranking → LLM Generation. Each arrow is labeled with the specific information loss at that transition. Below the pipeline, a red dashed line shows the quality inequality: quality(a) ≤ quality(R′) ≤ quality(R) ≤ quality(I) ≤ quality(E) ≤ quality(C) ≤ quality(D). A callout at Stage 1 reads: "Information destroyed here is invisible to every stage downstream."

**Figure 27.2: The Chunking Split — How Fixed-Size Boundaries Sever Conceptual Units**
A side-by-side comparison. Left panel: the original documentation section (~660 tokens) as a single block, with prose and code visually connected. A 512-token boundary line cuts through the transition zone. Right panel: two separate chunks — Chunk A (prose only) and Chunk B (code only) — with a red "severed" icon between them. Below, the retrieval result: Chunk A ranks #2 (cosine similarity 0.85), Chunk B ranks #14 (cosine similarity 0.62). The LLM receives only Chunk A. The invented `.set_backoff()` method appears in the output.

**Figure 27.3: The Embedding Space Geometry — Why Tuning Cannot Cross a Structural Gap**
A UMAP-style 2D projection showing: the query point (star) in the upper-left; a tight cluster of three prose chunks near it (similarity 0.79–0.85); a separate cluster of two code chunks in the lower-right (similarity 0.58–0.62); a statutory chunk isolated in the lower-left (similarity 0.31). Two concentric circles centered on the query: the inner circle (top-5 radius) captures only prose chunks; the outer circle (top-20 radius) barely reaches the code chunks. Gray dots fill the space between the query and the code cluster, representing the hundreds of intervening chunks. Annotations: "Ranked low — reachable by increasing k" for the code cluster, "Geometrically unreachable — no value of k helps" for the statutory chunk.

**Figure 27.4: The Structure–Logic–Implementation–Outcome Traversal**
A four-row diagnostic template, instantiated for the chunking stage. Row 1 (Structure): "Input: Document corpus → Output: Independent text segments." Row 2 (Logic): "Fixed-size windows cannot access document meaning. Severed concepts are unrecoverable." Row 3 (Implementation): "512-token window, no overlap, no structure awareness." Row 4 (Outcome): "Fluent but structurally incomplete answers. Failure persists after embedding/index/reranker changes." An arrow labeled "Diagnostic direction" points from Outcome back up to Structure.

**Figure 27.5: The Reranker's Ceiling — Precision Cannot Compensate for Recall**
Two panels. Left panel: "Chunk in candidate set at position 14." The reranker promotes it to position 2. Green checkmark. Right panel: "Chunk not in candidate set." The reranker has nothing to promote. Red X. Below both panels: the inequality *quality(R′) ≤ quality(R)* with the annotation "The reranker's ceiling is the retriever's recall."
