import math
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool

from local_agent.rag.chunking import load_and_split_documents, split_text
from local_agent.rag.tokenization import tokenize_for_bm25


def build_vector_store(embedding_model: str, persist_directory: Path):
    """Create a Chroma store, importing optional dependencies only when used."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=str(persist_directory), embedding_function=embeddings)


def bm25_scores(
    query: str,
    documents: list[Document],
    k1: float = 1.5,
    b: float = 0.75,
    *,
    tokenizer_mode: str = "accurate",
    stopwords_path: Path | None = None,
    user_dict_path: Path | None = None,
) -> list[float]:
    """Return BM25 scores aligned with the input documents."""

    def tokenize(text: str) -> list[str]:
        return tokenize_for_bm25(
            text,
            mode=tokenizer_mode,
            stopwords_path=stopwords_path,
            user_dict_path=user_dict_path,
        )

    query_terms = tokenize(query)
    if not query_terms or not documents:
        return [0.0 for _ in documents]

    tokenized_docs = [tokenize(document.page_content) for document in documents]
    doc_freq: Counter[str] = Counter()
    term_freqs: list[Counter[str]] = []
    lengths: list[int] = []
    for tokens in tokenized_docs:
        counts = Counter(tokens)
        term_freqs.append(counts)
        lengths.append(len(tokens))
        doc_freq.update(counts.keys())

    total_docs = len(documents)
    avg_len = sum(lengths) / total_docs if total_docs else 0.0
    if avg_len == 0:
        return [0.0 for _ in documents]

    scores: list[float] = []
    for counts, doc_len in zip(term_freqs, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            score += idf * (tf * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def reciprocal_rank_fusion(
    vector_hits: list[tuple[Document, float]],
    bm25_hits: list[tuple[Document, float]],
    top_k: int,
    rrf_k: int = 60,
    min_vector_score: float | None = None,
) -> list[dict[str, object]]:
    """Fuse vector and BM25 rankings into one result list."""
    fused: dict[str, dict[str, object]] = {}

    def key_for(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"{document.metadata.get('source', '')}:{document.metadata.get('chunk', -1)}"

    for rank, (document, score) in enumerate(vector_hits, start=1):
        key = key_for(document)
        item = fused.setdefault(
            key,
            {"document": document, "vector_score": 0.0, "bm25_score": 0.0, "score": 0.0},
        )
        item["vector_score"] = score
        item["score"] = float(item["score"]) + 1 / (rrf_k + rank)

    for rank, (document, score) in enumerate(bm25_hits, start=1):
        key = key_for(document)
        item = fused.setdefault(
            key,
            {"document": document, "vector_score": 0.0, "bm25_score": 0.0, "score": 0.0},
        )
        item["bm25_score"] = score
        item["score"] = float(item["score"]) + 1 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda item: float(item["score"]), reverse=True)
    if min_vector_score is not None:
        ranked = [
            item
            for item in ranked
            if float(item["bm25_score"]) > 0
            or float(item["vector_score"]) >= min_vector_score
        ]
    results: list[dict[str, object]] = []
    for item in ranked[:top_k]:
        document = item["document"]
        assert isinstance(document, Document)
        results.append(
            {
                "text": document.page_content,
                "source": document.metadata.get("source", ""),
                "chunk_id": document.metadata.get("chunk_id", ""),
                "page": document.metadata.get("page"),
                "header_path": document.metadata.get("header_path", ""),
                "score": item["score"],
                "vector_score": item["vector_score"],
                "bm25_score": item["bm25_score"],
            }
        )
    return results


def index_documents(
    knowledge_dir: Path,
    embedding_model: str,
    persist_directory: Path,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> int:
    """Build or update the local Chroma index and return the number of chunks added."""
    chunks = load_and_split_documents(knowledge_dir, chunk_size, chunk_overlap)
    if not chunks:
        return 0
    store = build_vector_store(embedding_model, persist_directory)
    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def retrieve_knowledge(
    query: str,
    embedding_model: str,
    persist_directory: Path,
    top_k: int = 4,
    knowledge_dir: Path | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    candidate_k: int = 20,
    min_vector_score: float = 0.3,
    tokenizer_mode: str = "accurate",
    stopwords_path: Path | None = None,
    user_dict_path: Path | None = None,
) -> list[dict[str, object]]:
    """Retrieve top-k chunks using vector search plus a local BM25 reranker."""
    candidate_k = max(candidate_k, top_k)
    store = build_vector_store(embedding_model, persist_directory)
    vector_results = store.similarity_search_with_relevance_scores(query, k=candidate_k)

    if knowledge_dir is None:
        return [
            {
                "text": document.page_content,
                "source": document.metadata.get("source", ""),
                "chunk_id": document.metadata.get("chunk_id", ""),
                "page": document.metadata.get("page"),
                "header_path": document.metadata.get("header_path", ""),
                "score": score,
                "vector_score": score,
                "bm25_score": 0.0,
            }
            for document, score in vector_results[:top_k]
        ]

    documents = load_and_split_documents(knowledge_dir, chunk_size, chunk_overlap)
    searchable_documents = [
        Document(
            page_content=(
                f"{document.metadata.get('source', '')} "
                f"{document.metadata.get('header_path', '')} "
                f"{document.page_content}"
            ),
            metadata=document.metadata,
        )
        for document in documents
    ]
    scores = bm25_scores(
        query,
        searchable_documents,
        tokenizer_mode=tokenizer_mode,
        stopwords_path=stopwords_path,
        user_dict_path=user_dict_path,
    )
    bm25_results = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:candidate_k]

    return reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k,
        min_vector_score=min_vector_score,
    )


def make_retrieval_tool(
    embedding_model: str,
    persist_directory: Path,
    top_k: int = 4,
    knowledge_dir: Path | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    candidate_k: int = 20,
    min_vector_score: float = 0.3,
    tokenizer_mode: str = "accurate",
    stopwords_path: Path | None = None,
    user_dict_path: Path | None = None,
):
    """Create a LangChain tool bound to the configured local vector store."""

    @tool
    def retrieve_local_knowledge(query: str) -> str:
        """Search the local Markdown, text, and PDF knowledge base and return sourced passages."""
        hits = retrieve_knowledge(
            query,
            embedding_model,
            persist_directory,
            top_k,
            knowledge_dir=knowledge_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            candidate_k=candidate_k,
            min_vector_score=min_vector_score,
            tokenizer_mode=tokenizer_mode,
            stopwords_path=stopwords_path,
            user_dict_path=user_dict_path,
        )
        if not hits:
            return "未检索到相关本地知识。"
        blocks = []
        for index, hit in enumerate(hits, start=1):
            location = f"source={hit['source']} chunk_id={hit['chunk_id']}"
            if hit.get("page") is not None:
                location += f" page={hit['page']}"
            if hit.get("header_path"):
                location += f" header={hit['header_path']}"
            blocks.append(
                f"[{index}] {location} score={hit['score']:.4f} "
                f"vector={hit['vector_score']:.4f} bm25={hit['bm25_score']:.4f}\n{hit['text']}"
            )
        return "\n\n".join(blocks)

    return retrieve_local_knowledge


__all__ = [
    "bm25_scores",
    "index_documents",
    "load_and_split_documents",
    "make_retrieval_tool",
    "reciprocal_rank_fusion",
    "retrieve_knowledge",
    "split_text",
    "tokenize_for_bm25",
]
