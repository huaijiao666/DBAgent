"""Encoding-safe console output shared by CLI and trace rendering."""

from __future__ import annotations

import sys
from typing import TextIO


def safe_print(
    text: object = "",
    *,
    stream: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print without letting a legacy terminal codec crash an agent run.

    Windows terminals may expose encodings such as GBK while providers can
    return text in any language. The original Unicode remains in model state;
    only terminal rendering falls back to replacement characters when needed.
    """

    target = stream or sys.stdout
    rendered = str(text)
    try:
        print(rendered, file=target, flush=flush)
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "ascii"
        compatible = rendered.encode(encoding, errors="replace").decode(
            encoding,
            errors="replace",
        )
        print(compatible, file=target, flush=flush)
