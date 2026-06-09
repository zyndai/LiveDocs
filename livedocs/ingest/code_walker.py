"""Walk CODE_DIR and yield indexable source files.

Drop any number of repos/folders into CODE_DIR (config.py). This module
recurses into all of them, skipping junk directories and non-source files, and
yields one record per file with its repo name, relative path, language, and
raw text. Build-time only -- nothing here runs per query.
"""
from pathlib import Path

from livedocs.config import (
    CODE_DIR,
    CODE_LANG_BY_EXT,
    CODE_IGNORE_DIRS,
    CODE_MAX_FILE_BYTES,
)


def _repo_name(rel_parts):
    """First path component under CODE_DIR is the repo/folder name."""
    return rel_parts[0] if len(rel_parts) > 1 else "(root)"


def walk_code(root=None):
    """Yield dicts: {repo, path (rel, posix), abs_path, lang, text}.

    Skips dirs in CODE_IGNORE_DIRS (and any dotdir), files whose extension is
    not in CODE_LANG_BY_EXT, oversized files, and anything that fails to decode
    as UTF-8 (binaries, bad encodings).
    """
    root = Path(root) if root is not None else CODE_DIR
    if not root.exists():
        raise SystemExit(
            f"CODE_DIR not found: {root}\n"
            f"Create it and drop your repos/folders inside, then re-run."
        )

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(root).parts
        # Skip if any parent dir is ignored or hidden.
        if any(p in CODE_IGNORE_DIRS or p.startswith(".") for p in rel_parts[:-1]):
            continue

        lang = CODE_LANG_BY_EXT.get(path.suffix.lower())
        if lang is None:
            continue

        try:
            if path.stat().st_size > CODE_MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if not text.strip():
            continue

        yield {
            "repo": _repo_name(rel_parts),
            "path": path.relative_to(root).as_posix(),
            "abs_path": str(path),
            "lang": lang,
            "text": text,
        }


if __name__ == "__main__":
    n = 0
    by_lang = {}
    for rec in walk_code():
        n += 1
        by_lang[rec["lang"]] = by_lang.get(rec["lang"], 0) + 1
        if n <= 20:
            print(f"  {rec['repo']:>16} | {rec['lang']:>10} | {rec['path']}")
    print(f"\n{n} files. By language: {by_lang}")
