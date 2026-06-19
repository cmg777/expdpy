"""A small stdout-capture context manager (several pyfixest helpers print to stdout)."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterator

__all__ = ["capture_stdout"]


@contextlib.contextmanager
def capture_stdout() -> Iterator[io.StringIO]:
    """Redirect ``sys.stdout`` into a buffer for the duration of the ``with`` block.

    Some pyfixest helpers (notably ``etable(type="md")``) print to stdout and return
    ``None``; this captures that text so it can be returned as a string instead.

    Yields
    ------
    io.StringIO
        The buffer; read its contents with ``.getvalue()`` after the block.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf
