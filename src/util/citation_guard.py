"""Streaming guard for markdown citation links in model output.

The retrieval context gives the model one ``### [title](link)`` header per
retrieved document and tells it to cite only those links. A model can still
invent a plausible URL or repeat the same citation on every bullet. Streamed
text has already reached the user by the time a whole answer could be
checked, so this guard validates each markdown link as it completes:

* a link that was not retrieved in this turn (or earlier in the conversation)
  is replaced by its plain link text;
* a link to an already-cited source is dropped or reduced to plain text.

The guard is a pure text transformer with no Azure or framework dependency.
Only markdown links and images are inspected; bare URLs, autolinks and
reference-style definitions are not.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

MODE_ENFORCE: Final = "enforce"
MODE_LOG_ONLY: Final = "log_only"
MODE_OFF: Final = "off"
VALID_MODES: Final = frozenset({MODE_ENFORCE, MODE_LOG_ONLY, MODE_OFF})
DEFAULT_MODE: Final = MODE_ENFORCE

# A link longer than this is not buffered further. Its opening bracket is
# emitted as ordinary text, which leaves the remainder as non-link text.
MAX_LINK_CHARS: Final = 4096

_LIST_PREFIX = re.compile(r"\s*(?:(?:[-*+\u2022>]|\d+[.)])\s*)*")
_TRAILING_SENTENCE_END: Final = ".!?;:"

_NEED_MORE: Final = object()
_NOT_A_LINK: Final = object()


def resolve_mode(value: object) -> str:
    """Return a valid mode. Unknown values fail closed to enforcement."""
    candidate = str(value or "").strip().lower() or DEFAULT_MODE
    if candidate in VALID_MODES:
        return candidate
    logger.warning(
        "[CitationGuard] Unknown CITATION_GUARD_MODE; using %s. Valid values: %s",
        DEFAULT_MODE,
        ", ".join(sorted(VALID_MODES)),
    )
    return DEFAULT_MODE


def canonical_url(url: str) -> str:
    """Normalize a link for comparison: lowercase scheme and host, no fragment."""
    candidate = url.strip()
    try:
        parts = urlsplit(candidate)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    except ValueError:
        return candidate


def _destination(inner: str) -> str:
    """Return the URL of a markdown link destination (drops an optional title)."""
    inner = inner.strip()
    if inner.startswith("<") and ">" in inner:
        return inner[1 : inner.index(">")].strip()
    return inner.split(None, 1)[0] if inner else ""


@dataclass
class GuardStats:
    links_seen: int = 0
    unretrieved: int = 0
    repeated: int = 0

    @property
    def altered(self) -> int:
        return self.unretrieved + self.repeated


class CitationGuard:
    """Stateful, chunk-boundary-safe sanitizer for one streamed answer."""

    def __init__(
        self,
        allowed_links: Callable[[], Iterable[str]],
        mode: str = DEFAULT_MODE,
    ) -> None:
        self._allowed_links = allowed_links
        self.mode = resolve_mode(mode)
        self.stats = GuardStats()
        self._pending = ""
        self._seen: set[str] = set()
        self._line_prefix = ""

    # -- public API ---------------------------------------------------------

    def feed(self, text: str) -> str:
        """Accept a streamed fragment; return the text that is safe to emit."""
        if self.mode == MODE_OFF:
            return text
        self._pending += text
        return self._drain(final=False)

    def flush(self) -> str:
        """Emit whatever is buffered once the stream has ended."""
        if self.mode == MODE_OFF:
            return ""
        return self._drain(final=True)

    def log_summary(self) -> None:
        if self.stats.altered:
            logger.warning(
                "[CitationGuard] mode=%s links=%d unretrieved=%d repeated=%d",
                self.mode,
                self.stats.links_seen,
                self.stats.unretrieved,
                self.stats.repeated,
            )

    # -- internals ----------------------------------------------------------

    def _drain(self, *, final: bool) -> str:
        buf = self._pending
        held = ""
        # A trailing "!" may start an image link whose "[" has not arrived yet.
        if not final and buf.endswith("!"):
            held, buf = "!", buf[:-1]

        out: list[str] = []
        pos = 0
        length = len(buf)
        while pos < length:
            idx = buf.find("[", pos)
            if idx == -1:
                self._emit(out, buf[pos:])
                pos = length
                break
            is_image = idx > pos and buf[idx - 1] == "!"
            start = idx - 1 if is_image else idx
            self._emit(out, buf[pos:start])
            parsed = self._parse_link(buf, idx, final=final)
            if parsed is _NEED_MORE:
                pos = start
                break
            if parsed is _NOT_A_LINK:
                self._emit(out, buf[start : idx + 1])
                pos = idx + 1
                continue
            end, label, inner = parsed  # type: ignore[misc]
            raw = buf[start:end]
            self._emit(out, self._decide(raw, label, inner, is_image))
            pos = end

        self._pending = buf[pos:] + held if pos < length else held
        return "".join(out)

    @staticmethod
    def _parse_link(buf: str, idx: int, *, final: bool):
        """Parse ``[label](destination)`` at ``idx`` or say what is missing."""
        length = len(buf)
        limit = idx + MAX_LINK_CHARS

        def incomplete(scanned_to: int):
            if final or scanned_to - idx >= MAX_LINK_CHARS:
                return _NOT_A_LINK
            return _NEED_MORE

        depth = 0
        i = idx
        while i < length:
            if i >= limit:
                return _NOT_A_LINK
            char = buf[i]
            if char == "\\":
                i += 2
                continue
            if char == "\n" and buf.startswith("\n", i + 1):
                return _NOT_A_LINK
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        else:
            return incomplete(length)
        label_end = i
        if label_end + 1 >= length:
            return incomplete(length)
        if buf[label_end + 1] != "(":
            return _NOT_A_LINK

        depth = 1
        j = label_end + 2
        while j < length:
            if j >= limit:
                return _NOT_A_LINK
            char = buf[j]
            if char == "\\":
                j += 2
                continue
            if char == "\n":
                return _NOT_A_LINK
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return j + 1, buf[idx + 1 : label_end], buf[label_end + 2 : j]
            j += 1
        return incomplete(length)

    def _decide(self, raw: str, label: str, inner: str, is_image: bool) -> str:
        """Return the text to emit in place of one complete markdown link."""
        destination = _destination(inner)
        self.stats.links_seen += 1
        if destination.startswith("#"):
            return raw
        canonical = canonical_url(destination)
        allowed = {canonical_url(link) for link in self._allowed_links() if link}
        if canonical not in allowed:
            self.stats.unretrieved += 1
            return raw if self.mode == MODE_LOG_ONLY else label
        if canonical in self._seen:
            self.stats.repeated += 1
            if self.mode == MODE_LOG_ONLY:
                return raw
            return self._repeat_replacement(label)
        self._seen.add(canonical)
        return raw

    def _repeat_replacement(self, label: str) -> str:
        """Drop a trailing repeat citation; keep the label where the sentence needs it."""
        prefix = self._line_prefix
        if _LIST_PREFIX.fullmatch(prefix):
            return label
        last = prefix.rstrip()[-1:]
        if last and last in _TRAILING_SENTENCE_END:
            return ""
        return label

    def _emit(self, out: list[str], text: str) -> None:
        if not text:
            return
        out.append(text)
        newline = text.rfind("\n")
        self._line_prefix = (
            text[newline + 1 :] if newline != -1 else self._line_prefix + text
        )[-200:]


def extract_link_destinations(text: str) -> list[str]:
    """Return the destinations of the markdown links in ``text``.

    Uses the guard's own scanner so both agree on what a link is. Intended for
    earlier assistant turns, which were validated before they were stored.
    """
    text = text or ""
    found: list[str] = []
    pos = 0
    while True:
        idx = text.find("[", pos)
        if idx == -1:
            return found
        parsed = CitationGuard._parse_link(text, idx, final=True)
        if parsed is _NOT_A_LINK:
            pos = idx + 1
            continue
        end, _label, inner = parsed  # type: ignore[misc]
        found.append(_destination(inner))
        pos = end
