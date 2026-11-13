"""One token counter shared by every method.

Budgets and compression numbers only mean something if all methods are
measured with the same tokenizer. We use tiktoken's cl100k_base because it is
fast, deterministic and not tied to any of the models under test.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Number of tokens in ``text``. Special-token strings are treated as plain text."""
    if not text:
        return 0
    if len(text) < _CACHE_MIN_CHARS:
        return len(_encoding().encode_ordinary(text))
    return _count_long(text)


# Rendered files get counted more than once (by the selector, then by the
# metrics), so remember recent long texts. Short strings are cheap to redo.
_CACHE_MIN_CHARS = 400


@lru_cache(maxsize=512)
def _count_long(text: str) -> int:
    return len(_encoding().encode_ordinary(text))


def count_tokens_batch(texts: list[str]) -> list[int]:
    """Token counts for many strings.

    A plain loop on purpose: tiktoken's own batch call starts a thread pool per
    call, which is far slower than this for many short strings such as lines.
    """
    encode = _encoding().encode_ordinary
    return [len(encode(t)) if t else 0 for t in texts]
