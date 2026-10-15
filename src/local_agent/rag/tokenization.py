import re
from functools import lru_cache
from pathlib import Path

import jieba

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")


@lru_cache(maxsize=16)
def load_word_list(path: str | None) -> frozenset[str]:
    """Load one UTF-8 word per line, ignoring blanks and comments."""
    if not path:
        return frozenset()
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Word list does not exist: {file_path}")
    return frozenset(
        line.strip().lower()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@lru_cache(maxsize=8)
def build_tokenizer(user_dict_path: str | None = None) -> jieba.Tokenizer:
    """Build an isolated jieba tokenizer with an optional project dictionary."""
    tokenizer = jieba.Tokenizer()
    if user_dict_path:
        path = Path(user_dict_path)
        if not path.exists():
            raise FileNotFoundError(f"jieba user dictionary does not exist: {path}")
        tokenizer.load_userdict(str(path))
    return tokenizer


def tokenize_for_bm25(
    text: str,
    *,
    mode: str = "accurate",
    stopwords_path: Path | str | None = None,
    user_dict_path: Path | str | None = None,
) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25 with jieba."""
    if mode not in {"accurate", "search"}:
        raise ValueError("jieba mode must be 'accurate' or 'search'")

    tokenizer = build_tokenizer(str(user_dict_path) if user_dict_path else None)
    raw_tokens = (
        tokenizer.lcut_for_search(text) if mode == "search" else tokenizer.lcut(text, cut_all=False)
    )
    stopwords = load_word_list(str(stopwords_path) if stopwords_path else None)
    tokens: list[str] = []
    for raw_token in raw_tokens:
        for match in TOKEN_RE.finditer(raw_token):
            token = match.group(0).lower()
            if token and token not in stopwords:
                tokens.append(token)
    return tokens
