"""Heading-aware markdown chunker (local copy for ragpipline/).

Two-pass design:
  Pass 1 (parse_sections / parse_docs_tree) -- structural split by headings,
    respecting code fences so '# ...' inside code doesn't count as a heading.
  Pass 2 (chunk_sections) -- merge tiny adjacent same-heading sections (bounded
    so cascades stop at MERGE_STOP_TOKENS), then split any oversized sections
    by indivisible blocks (code/tables kept whole), then add overlap between
    pieces of a split section (never across heading boundaries).

Identical logic to the root chunker.py -- duplicated here so ragpipline/ stays
self-contained.
"""
import re
from pathlib import Path

import tiktoken

from livedocs.config import (
    MAX_TOKENS, MIN_TOKENS, OVERLAP_TOKENS, MERGE_STOP_TOKENS,
    TOKENIZER_ENCODING, IGNORE_DIRS,
)

enc = tiktoken.get_encoding(TOKENIZER_ENCODING)

_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def count_tokens(text):
    return len(enc.encode(text))


# -------- Pass 1: structural parse --------

def parse_sections(filepath, source_label=None, product_area=None):
    sections = []
    heading = subheading = None
    buffer = []
    in_code = False
    source = source_label if source_label is not None else str(filepath)

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            sections.append({
                "source": source,
                "product_area": product_area,
                "heading": heading,
                "subheading": subheading,
                "content": content,
            })

    with open(filepath, encoding="utf-8") as f:
        raw = f.read()
    raw = _FRONTMATTER_RE.sub("", raw, count=1)

    for line in raw.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            buffer.append(line)
            continue

        if not in_code and line.startswith("# "):
            flush(); buffer = []
            heading, subheading = line[2:].strip(), None
        elif not in_code and line.startswith("## "):
            flush(); buffer = []
            subheading = line[3:].strip()
        elif not in_code and line.startswith("### "):
            flush(); buffer = []
            subheading = line[4:].strip()
        else:
            buffer.append(line)

    flush()
    return sections


def walk_docs(root):
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        parts = path.relative_to(root).parts
        if any(p in IGNORE_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        yield path


def parse_docs_tree(root):
    root = Path(root)
    all_sections = []
    for path in walk_docs(root):
        rel = path.relative_to(root).as_posix()
        product_area = rel.split("/")[0] if "/" in rel else None
        all_sections.extend(parse_sections(path, source_label=rel, product_area=product_area))
    return all_sections


# -------- Pass 2: size fixes --------

def split_blocks(content):
    lines = content.split("\n")
    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            start = i
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                i += 1
            if i < n:
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            start = i
            while i < n and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        if stripped == "":
            i += 1
            continue

        start = i
        while i < n:
            s = lines[i].strip()
            if s == "" or s.startswith("```") or (s.startswith("|") and s.endswith("|")):
                break
            i += 1
        blocks.append("\n".join(lines[start:i]))

    return blocks


def tail_tokens(text, n_tokens):
    tokens = enc.encode(text)
    if len(tokens) <= n_tokens:
        return text
    return enc.decode(tokens[-n_tokens:])


def split_large_section(section):
    content = section["content"]
    if count_tokens(content) <= MAX_TOKENS:
        return [content]

    blocks = split_blocks(content)
    chunks = []
    current = []
    current_tokens = 0

    for block in blocks:
        btok = count_tokens(block)
        if btok > MAX_TOKENS:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            print(f"  ! oversized block ({btok} tok) kept whole in [{section['heading']} > {section['subheading']}]")
            chunks.append(block)
            continue

        if current_tokens + btok > MAX_TOKENS and current:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(block)
        current_tokens += btok

    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) > 1:
        with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = tail_tokens(chunks[i - 1], OVERLAP_TOKENS)
            with_overlap.append(overlap + "\n\n" + chunks[i])
        chunks = with_overlap

    return chunks


def _format_subheading(prev_sub, new_sub, extra_count):
    if not new_sub or new_sub == prev_sub:
        return prev_sub
    if not prev_sub:
        return new_sub
    if extra_count <= 2:
        return f"{prev_sub} + {new_sub}"
    base = prev_sub.split(" [+")[0].split(" + ")[0]
    return f"{base} [+{extra_count} subsections]"


def merge_small_sections(sections):
    merged = []
    merge_counts = []
    for sec in sections:
        tok = count_tokens(sec["content"])
        if (tok < MIN_TOKENS
                and merged
                and merged[-1]["heading"] == sec["heading"]
                and count_tokens(merged[-1]["content"]) < MERGE_STOP_TOKENS):
            prev = merged[-1]
            prev["content"] = prev["content"] + "\n\n" + sec["content"]
            merge_counts[-1] += 1
            prev["subheading"] = _format_subheading(prev["subheading"], sec["subheading"], merge_counts[-1])
            continue
        merged.append(dict(sec))
        merge_counts.append(0)

    out = []
    i = 0
    while i < len(merged):
        sec = merged[i]
        tok = count_tokens(sec["content"])
        if (tok < MIN_TOKENS
                and not out
                and i + 1 < len(merged)
                and merged[i + 1]["heading"] == sec["heading"]):
            nxt = dict(merged[i + 1])
            nxt["content"] = sec["content"] + "\n\n" + nxt["content"]
            if sec["subheading"] and sec["subheading"] != nxt["subheading"]:
                nxt["subheading"] = f"{sec['subheading']} + {nxt['subheading']}" if nxt["subheading"] else sec["subheading"]
            merged[i + 1] = nxt
            i += 1
            continue
        out.append(sec)
        i += 1

    return out


def chunk_sections(sections):
    sections = merge_small_sections(sections)
    chunks = []
    for sec in sections:
        pieces = split_large_section(sec)
        total = len(pieces)
        for idx, piece in enumerate(pieces):
            chunks.append({
                "source": sec["source"],
                "product_area": sec.get("product_area"),
                "heading": sec["heading"],
                "subheading": sec["subheading"],
                "content": piece,
                "chunk_index": idx,
                "chunk_total": total,
                "token_count": count_tokens(piece),
            })
    return chunks


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "test.md"
    target_path = Path(target)

    if target_path.is_dir():
        raw = parse_docs_tree(target_path)
    else:
        raw = parse_sections(target_path, source_label=target_path.name)
    print(f"Pass 1: {len(raw)} sections from {target}\n")

    chunks = chunk_sections(raw)
    print(f"\nPass 2: {len(chunks)} final chunks\n")

    for c in chunks:
        tag = f"[{c['heading']} > {c['subheading']}]"
        suffix = f" (part {c['chunk_index'] + 1}/{c['chunk_total']})" if c["chunk_total"] > 1 else ""
        print(f"{tag}{suffix} -- {c['token_count']} tok")
        print(c["content"][:200].replace("\n", " "))
        print("---")
