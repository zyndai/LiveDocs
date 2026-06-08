"""Shared serialization helpers used by both code_pipeline and app."""


def doc_to_source_dict(d, with_score=True):
    m = d.meta or {}
    is_code = bool(m.get("node_id") or m.get("repo"))
    base = {
        "type": "code" if is_code else "doc",
        "score": float(d.score) if (with_score and d.score is not None) else None,
    }
    if is_code:
        base.update({
            "repo": m.get("repo"),
            "file": m.get("file"),
            "symbol": m.get("symbol"),
            "symbol_type": m.get("symbol_type"),
            "start_line": m.get("start_line"),
            "end_line": m.get("end_line"),
        })
    else:
        base.update({
            "source": m.get("source"),
            "heading": m.get("heading"),
            "subheading": m.get("subheading"),
            "product_area": m.get("product_area"),
        })
    return base


def docs_to_source_dicts(docs, with_score=True):
    return [doc_to_source_dict(d, with_score=with_score) for d in docs]
