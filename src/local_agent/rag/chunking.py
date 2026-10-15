import re
from hashlib import sha256
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")


def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """Split plain text with recursive, paragraph-aware separators."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    if not text.strip():
        return []
    splitter = _recursive_splitter(chunk_size, chunk_overlap)
    return [part for part in splitter.split_text(text) if part.strip()]


def load_and_split_documents(
    root: Path, chunk_size: int = 600, chunk_overlap: int = 100
) -> list[Document]:
    """Load Markdown, text, and PDF files into stable retrieval chunks."""
    validate_chunk_settings(chunk_size, chunk_overlap)
    chunks: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        source = path.relative_to(root).as_posix()
        if source == "README.md":
            continue
        if suffix == ".md":
            source_chunks = _split_markdown(path, source, chunk_size, chunk_overlap)
        elif suffix == ".txt":
            source_chunks = _split_plain_text(path, source, chunk_size, chunk_overlap)
        elif suffix == ".pdf":
            source_chunks = _split_pdf(path, source, chunk_size, chunk_overlap)
        else:
            continue
        chunks.extend(_finalize_metadata(source_chunks, source))
    return chunks


def _recursive_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )


def _split_markdown(path: Path, source: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    for section in sections:
        section.metadata["source"] = source
        section.metadata["header_path"] = " > ".join(
            str(section.metadata[key]) for key in ("h1", "h2", "h3") if section.metadata.get(key)
        )
    return _recursive_splitter(chunk_size, chunk_overlap).split_documents(sections)


def _split_plain_text(
    path: Path, source: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return _recursive_splitter(chunk_size, chunk_overlap).create_documents(
        [text], metadatas=[{"source": source, "header_path": ""}]
    )


def _split_pdf(path: Path, source: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    pages: list[Document] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "page": page_number, "header_path": ""},
                )
            )
    return _recursive_splitter(chunk_size, chunk_overlap).split_documents(pages)


def _finalize_metadata(documents: list[Document], source: str) -> list[Document]:
    for index, document in enumerate(documents):
        normalized = _normalize_content(document.page_content)
        header_path = str(document.metadata.get("header_path", ""))
        content_hash = sha256(normalized.encode("utf-8")).hexdigest()
        identity = f"{source}\x1f{header_path}\x1f{normalized}"
        document.metadata.update(
            {
                "source": source,
                "chunk": index,
                "chunk_index": index,
                "chunk_id": sha256(identity.encode("utf-8")).hexdigest()[:16],
                "content_hash": content_hash,
                "header_path": header_path,
            }
        )
    return documents


def _normalize_content(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
