from __future__ import annotations

from pathlib import Path


def format_display_path(path: str | Path, *, trailing_slash: bool = False) -> str:
    """Return a reproducible path display, preferring relative paths."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(Path.cwd())
        except ValueError:
            return candidate.name

    text = candidate.as_posix()
    if text in ("", "."):
        text = "."
    elif not text.startswith("."):
        text = f"./{text}"

    if trailing_slash and not text.endswith("/"):
        text += "/"
    return text
