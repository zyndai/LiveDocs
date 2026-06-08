"""Shared serialization helpers used by both code_pipeline and app."""
from config import REPO_GITHUB_URLS, REPO_GITHUB_BRANCH


def _github_url(repo: str, file: str, start_line: int | None, end_line: int | None) -> str | None:
    """Build a GitHub blob URL for a code chunk, or None if repo not mapped."""
    base = REPO_GITHUB_URLS.get(repo)
    if not base:
        return None
    branch = (
        REPO_GITHUB_BRANCH.get(repo, "main")
        if isinstance(REPO_GITHUB_BRANCH, dict)
        else REPO_GITHUB_BRANCH
    )
    # file includes the repo prefix (e.g. "AgentDNS/internal/mesh/federated.go");
    # strip it to get the within-repo path.
    within_repo = file[len(repo) + 1:] if file.startswith(repo + "/") else file
    url = f"{base.rstrip('/')}/blob/{branch}/{within_repo}"
    if start_line is not None:
        url += f"#L{start_line}"
        if end_line is not None and end_line != start_line:
            url += f"-L{end_line}"
    return url


def doc_to_source_dict(d, with_score=True):
    m = d.meta or {}
    is_code = bool(m.get("node_id") or m.get("repo"))
    base = {
        "type": "code" if is_code else "doc",
        "score": float(d.score) if (with_score and d.score is not None) else None,
    }
    if is_code:
        repo = m.get("repo")
        file = m.get("file")
        start_line = m.get("start_line")
        end_line = m.get("end_line")
        base.update({
            "repo": repo,
            "file": file,
            "symbol": m.get("symbol"),
            "symbol_type": m.get("symbol_type"),
            "start_line": start_line,
            "end_line": end_line,
            "github_url": _github_url(repo, file, start_line, end_line) if repo and file else None,
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
