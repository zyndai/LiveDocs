"""Tree-sitter code chunker for Python / JS / TS / Go.

One chunk per top-level function, method, class/struct/interface (oversized
classes are split into a header chunk + per-method chunks; oversized functions
are split into overlapping line windows). Each chunk also carries the call and
import edges tree-sitter found in its subtree -- code_graph.py turns those into
the runtime graph.

Build-time only. Returns plain dicts; index.py converts them to Haystack
Documents and code_graph.py consumes the same dicts for edges.
"""
import tiktoken
from tree_sitter import Parser
from tree_sitter_language_pack import get_language

from livedocs.config import (
    TOKENIZER_ENCODING,
    CODE_MAX_TOKENS,
    CODE_OVERLAP_TOKENS,
)

enc = tiktoken.get_encoding(TOKENIZER_ENCODING)


def count_tokens(text):
    return len(enc.encode(text))


# --- Per-language node-type maps -------------------------------------------
# Definition nodes that become their own chunk, mapped to a symbol_type label.
DEF_TYPES = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
    },
    "typescript": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "tsx": {
        "function_declaration": "function",
        "method_definition": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "enum_declaration": "enum",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
    },
}

# Node types that contain methods we want to break out separately when the
# enclosing definition is too big to keep whole.
CLASS_LIKE = {"class_definition", "class_declaration"}

# Call-expression node types per language.
CALL_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression", "new_expression"},
    "typescript": {"call_expression", "new_expression"},
    "tsx": {"call_expression", "new_expression"},
    "go": {"call_expression"},
}

# Import-statement node types per language.
IMPORT_TYPES = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "tsx": {"import_statement"},
    "go": {"import_declaration"},
}

_PARSERS = {}


def _parser(lang):
    if lang not in _PARSERS:
        _PARSERS[lang] = Parser(get_language(lang))
    return _PARSERS[lang]


def _text(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _name_of(node, src):
    """Best-effort symbol name: the 'name' field, else first identifier child."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, src)
    # Go type_declaration wraps a type_spec that holds the name.
    for child in node.children:
        if child.type in ("type_spec", "var_spec"):
            n = child.child_by_field_name("name")
            if n is not None:
                return _text(n, src)
    for child in node.children:
        if child.type == "identifier":
            return _text(child, src)
    return "<anonymous>"


def _callee_name(call_node, src):
    """Extract the called symbol's bare name from a call/new node."""
    fn = call_node.child_by_field_name("function")
    if fn is None:
        # new_expression uses 'constructor'; Go selector handled below.
        fn = call_node.child_by_field_name("constructor")
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn, src)
    # attribute (py) / member_expression (js/ts) / selector_expression (go)
    prop = (
        fn.child_by_field_name("attribute")
        or fn.child_by_field_name("property")
        or fn.child_by_field_name("field")
    )
    if prop is not None:
        return _text(prop, src)
    # Fallback: last identifier descendant.
    last = None
    for d in _descendants(fn):
        if d.type in ("identifier", "field_identifier", "property_identifier"):
            last = d
    return _text(last, src) if last is not None else None


def _descendants(node):
    stack = list(node.children)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _collect_calls(node, src, lang):
    call_types = CALL_TYPES.get(lang, set())
    names = []
    seen = set()
    for d in _descendants(node):
        if d.type in call_types:
            name = _callee_name(d, src)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _leading_comment(node, src, lang):
    """Grab an immediately-preceding comment/docstring as chunk preamble."""
    prev = node.prev_sibling
    parts = []
    # Walk back over contiguous comment siblings.
    while prev is not None and prev.type in ("comment", "line_comment", "block_comment"):
        parts.append(_text(prev, src))
        prev = prev.prev_sibling
    parts.reverse()
    return "\n".join(parts)


def _split_oversized(text, max_tokens, overlap_tokens):
    """Split a too-big chunk into overlapping line windows (keeps it indexable)."""
    if count_tokens(text) <= max_tokens:
        return [text]
    lines = text.split("\n")
    chunks, cur, cur_tok = [], [], 0
    for line in lines:
        lt = count_tokens(line) + 1
        if cur_tok + lt > max_tokens and cur:
            chunks.append("\n".join(cur))
            # Overlap: carry the tail lines into the next window.
            carry, carry_tok = [], 0
            for prev_line in reversed(cur):
                pt = count_tokens(prev_line) + 1
                if carry_tok + pt > overlap_tokens:
                    break
                carry.insert(0, prev_line)
                carry_tok += pt
            cur, cur_tok = list(carry), carry_tok
        cur.append(line)
        cur_tok += lt
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _method_nodes(class_node, lang):
    """Method/function definition nodes directly inside a class-like node."""
    defs = DEF_TYPES.get(lang, {})
    out = []
    # Body is usually a 'block' / 'class_body' child holding the members.
    bodies = [c for c in class_node.children if c.type in ("block", "class_body", "field_declaration_list")]
    bodies = bodies or [class_node]
    for body in bodies:
        for child in body.children:
            if child.type in defs and child.type not in CLASS_LIKE:
                out.append(child)
    return out


def _make_chunk(file_rec, node, src, lang, symbol_type, name_override=None):
    name = name_override or _name_of(node, src)
    preamble = _leading_comment(node, src, lang)
    body = _text(node, src)
    content = f"{preamble}\n{body}".strip() if preamble else body
    return {
        "repo": file_rec["repo"],
        "file": file_rec["path"],
        "lang": lang,
        "symbol": name,
        "symbol_type": symbol_type,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "calls": _collect_calls(node, src, lang),
        "content": content,
    }


def _class_header(file_rec, class_node, methods, src, lang):
    """Class declaration up to its first method -- signature + class-level fields."""
    if methods:
        header_end = min(m.start_byte for m in methods)
    else:
        header_end = class_node.end_byte
    text = src[class_node.start_byte:header_end].decode("utf-8", errors="replace").rstrip()
    return {
        "repo": file_rec["repo"],
        "file": file_rec["path"],
        "lang": lang,
        "symbol": _name_of(class_node, src),
        "symbol_type": "class",
        "start_line": class_node.start_point[0] + 1,
        "end_line": class_node.end_point[0] + 1,
        "calls": [],
        "content": text,
    }


def _imports_chunk(file_rec, root, src, lang):
    """Collect all import statements into one file-level chunk + edge list."""
    import_types = IMPORT_TYPES.get(lang, set())
    nodes = [n for n in _descendants(root) if n.type in import_types]
    if not nodes:
        return None, []
    texts, modules = [], []
    for n in nodes:
        texts.append(_text(n, src))
        # Module names: pull string/identifier descendants as a cheap signal.
        for d in _descendants(n):
            if d.type in ("dotted_name", "identifier", "string", "interpreted_string_literal", "string_fragment"):
                modules.append(_text(d, src).strip("\"'`"))
    chunk = {
        "repo": file_rec["repo"],
        "file": file_rec["path"],
        "lang": lang,
        "symbol": "<imports>",
        "symbol_type": "imports",
        "start_line": min(n.start_point[0] for n in nodes) + 1,
        "end_line": max(n.end_point[0] for n in nodes) + 1,
        "calls": [],
        "content": "\n".join(texts),
    }
    # Dedupe (order-preserving), drop empties.
    modules = [m for m in dict.fromkeys(modules) if m]
    return chunk, modules


def chunk_file(file_rec):
    """Chunk one file record from code_walker.walk_code().

    Returns (chunks, imports) where imports is the module-name list for graph
    edges. Each chunk is a dict ready for index.build_documents().
    """
    lang = file_rec["lang"]
    if lang not in DEF_TYPES:
        return [], []
    src = file_rec["text"].encode("utf-8")
    tree = _parser(lang).parse(src)
    root = tree.root_node
    defs = DEF_TYPES[lang]

    raw_chunks = []

    imports_chunk, modules = _imports_chunk(file_rec, root, src, lang)
    if imports_chunk is not None:
        raw_chunks.append(imports_chunk)

    # Walk the tree; emit a chunk per definition. For class-like nodes, decide
    # whether to keep whole or split into header + methods based on size.
    def visit(node, inside_class):
        for child in node.children:
            ctype = child.type
            if ctype in defs:
                stype = defs[ctype]
                if ctype in CLASS_LIKE:
                    methods = _method_nodes(child, lang)
                    whole = _text(child, src)
                    if count_tokens(whole) <= CODE_MAX_TOKENS or not methods:
                        raw_chunks.append(_make_chunk(file_rec, child, src, lang, stype))
                    else:
                        raw_chunks.append(_class_header(file_rec, child, methods, src, lang))
                        for m in methods:
                            raw_chunks.append(_make_chunk(file_rec, m, src, lang, "method"))
                    # Don't double-descend into already-emitted methods.
                    continue
                else:
                    # Standalone function or top-level method.
                    if not inside_class:
                        raw_chunks.append(_make_chunk(file_rec, child, src, lang, stype))
                        continue
            visit(child, inside_class or ctype in CLASS_LIKE)

    visit(root, False)

    # Size-fix: split any oversized chunk into overlapping windows, preserving meta.
    final = []
    for c in raw_chunks:
        pieces = _split_oversized(c["content"], CODE_MAX_TOKENS, CODE_OVERLAP_TOKENS)
        total = len(pieces)
        for idx, piece in enumerate(pieces):
            nc = dict(c)
            nc["content"] = piece
            nc["chunk_index"] = idx
            nc["chunk_total"] = total
            nc["token_count"] = count_tokens(piece)
            final.append(nc)

    return final, modules


if __name__ == "__main__":
    import sys
    from code_walker import walk_code

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    shown = 0
    total_chunks = 0
    for rec in walk_code():
        chunks, modules = chunk_file(rec)
        total_chunks += len(chunks)
        if shown < limit and chunks:
            shown += 1
            print(f"\n=== {rec['repo']}/{rec['path']} ({rec['lang']}) "
                  f"-> {len(chunks)} chunks, imports={modules[:5]} ===")
            for c in chunks:
                print(f"  [{c['symbol_type']:>9}] {c['symbol']:<24} "
                      f"L{c['start_line']}-{c['end_line']} "
                      f"{c['token_count']}tok calls={c['calls'][:4]}")
    print(f"\nTotal: {total_chunks} chunks")