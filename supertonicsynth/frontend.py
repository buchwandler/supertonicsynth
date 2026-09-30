from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from unicodedata import normalize

import numpy as np

from .config import AVAILABLE_LANGUAGES
from .errors import InvalidLanguageError, InvalidRequestError

_EMOJI_PATTERN = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
    "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
    flags=re.UNICODE,
)
_SYMBOL_REPLACEMENTS = {
    "\u2013": "-",
    "\u2011": "-",
    "\u2014": "-",
    "\u00af": " ",
    "_": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u00b4": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}
_SPECIAL_SYMBOLS_PATTERN = re.compile(r"[♥☆♡©\\]")
_DUPLICATE_QUOTES_PATTERN = re.compile(r'(["\'`])\1+')
_WHITESPACE_PATTERN = re.compile(r"\s+")
_ENDING_PUNCTUATION_PATTERN = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


@dataclass(frozen=True, slots=True)
class FrontendBatch:
    text: str
    text_ids: np.ndarray
    text_mask: np.ndarray


class SupertonicFrontend:
    """Archive-compatible Supertonic-3 Unicode frontend."""

    def __init__(self, unicode_indexer_path: str | Path) -> None:
        path = Path(unicode_indexer_path)
        try:
            indexer = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidRequestError(f"Could not read Unicode indexer {path}: {exc}") from exc
        if not isinstance(indexer, list) or not indexer:
            raise InvalidRequestError("Unicode indexer must be a non-empty list")
        self.indexer = indexer
        self.supported_chars = {
            chr(i) for i, token in enumerate(indexer) if isinstance(token, int) and token >= 0
        }

    @staticmethod
    def _normalize_symbols(text: str) -> str:
        for old, new in _SYMBOL_REPLACEMENTS.items():
            text = text.replace(old, new)
        return text

    def preprocess(self, text: str, language: str | None = None) -> str:
        if not isinstance(text, str) or not text.strip():
            raise InvalidRequestError("text cannot be empty")
        if language is not None and language not in AVAILABLE_LANGUAGES:
            raise InvalidLanguageError(
                f"Unsupported language {language!r}; expected one of {', '.join(AVAILABLE_LANGUAGES)}"
            )
        text = normalize("NFKD", text)
        text = _EMOJI_PATTERN.sub("", text)
        text = self._normalize_symbols(text)
        text = _SPECIAL_SYMBOLS_PATTERN.sub("", text)
        text = (
            text.replace("@", " at ")
            .replace("e.g.,", "for example, ")
            .replace("i.e.,", "that is, ")
        )
        for before, after in (
            (" ,", ","),
            (" .", "."),
            (" !", "!"),
            (" ?", "?"),
            (" ;", ";"),
            (" :", ":"),
            (" '", "'"),
        ):
            text = text.replace(before, after)
        text = _DUPLICATE_QUOTES_PATTERN.sub(r"\1", text)
        text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        if not text:
            raise InvalidRequestError("text is empty after normalization")
        if not _ENDING_PUNCTUATION_PATTERN.search(text):
            text += "."
        if language is not None:
            text = f"<{language}>{text}</{language}>"
        return text

    def encode(self, text: str, language: str | None = None) -> FrontendBatch:
        normalized = self.preprocess(text, language)
        ids: list[int] = []
        unsupported: set[str] = set()
        for char in normalized:
            value = ord(char)
            if value >= len(self.indexer):
                unsupported.add(char)
                continue
            token = self.indexer[value]
            if not isinstance(token, int) or token < 0:
                unsupported.add(char)
                continue
            ids.append(token)
        if unsupported:
            rendered = " ".join(repr(v) for v in sorted(unsupported))
            raise InvalidRequestError(f"unsupported character(s): {rendered}")
        text_ids = np.asarray(ids, dtype=np.int64)
        text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
        return FrontendBatch(text=normalized, text_ids=text_ids, text_mask=text_mask)

    def validate_text(self, text: str) -> tuple[bool, tuple[str, ...]]:
        unsupported: set[str] = set()
        for char in set(text):
            try:
                normalized = self.preprocess(char, None)
            except InvalidRequestError:
                continue
            for value in normalized:
                if value not in self.supported_chars:
                    unsupported.add(char)
        return not unsupported, tuple(sorted(unsupported))
