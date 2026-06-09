"""Source management: add, remove, clone GitHub repos, register uploads."""
import shutil
import subprocess
import uuid
from pathlib import Path

from livedocs.config import PROJECT_ROOT

CODE_DIR = PROJECT_ROOT / "code"
DOCS_DIR = PROJECT_ROOT / "docs"


def _corpus_dir(corpus: str) -> Path:
    return CODE_DIR if corpus == "code" else DOCS_DIR


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


def add_github(name: str, url: str, corpus: str, branch: str = "main",
               github_base: str = "", token: str = "") -> dict:
    """Register a GitHub source. Does not clone immediately (jobs.py does that)."""
    if not github_base:
        github_base = url.rstrip("/")
        if github_base.endswith(".git"):
            github_base = github_base[:-4]

    source = {
        "id": _new_id(),
        "kind": "github",
        "corpus": corpus,
        "name": name,
        "url": url,
        "branch": branch,
        "github_base": github_base,
        "token": token,
        "status": "pending",
        "error": "",
        "counts": {},
    }
    _upsert_source(source)
    return source


def add_local(name: str, path: str, corpus: str) -> dict:
    """Register an existing local directory as a source."""
    source = {
        "id": _new_id(),
        "kind": "local",
        "corpus": corpus,
        "name": name,
        "url": path,
        "branch": "",
        "github_base": "",
        "token": "",
        "status": "pending",
        "error": "",
        "counts": {},
    }
    _upsert_source(source)
    return source


def add_upload(name: str, corpus: str, file_bytes: bytes, filename: str) -> dict:
    """Accept an uploaded zip or .md file and extract/save to the corpus dir."""
    import zipfile, io

    target_dir = _corpus_dir(corpus) / name
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            zf.extractall(target_dir)
    elif filename.endswith(".md"):
        (target_dir / filename).write_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Upload .zip or .md files.")

    source = {
        "id": _new_id(),
        "kind": "upload",
        "corpus": corpus,
        "name": name,
        "url": str(target_dir),
        "branch": "",
        "github_base": "",
        "token": "",
        "status": "pending",
        "error": "",
        "counts": {},
    }
    _upsert_source(source)
    return source


def clone_source(source: dict, log=print) -> None:
    """Clone or pull a GitHub source into code/ or docs/. Updates source status in-place."""
    from livedocs.settings import save_settings, get_settings

    src_id = source["id"]
    corpus = source["corpus"]
    name = source["name"]
    url = source["url"]
    branch = source.get("branch", "main")
    token = source.get("token", "")

    target = _corpus_dir(corpus) / name

    # Inject token into URL if provided
    clone_url = url
    if token and "github.com" in url:
        clone_url = url.replace("https://", f"https://{token}@")

    _update_source_status(src_id, "cloning")

    try:
        if target.exists() and (target / ".git").exists():
            log(f"  Pulling {name} (branch {branch})...")
            subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                check=True, capture_output=True, text=True,
            )
        else:
            log(f"  Cloning {url} -> {target} (branch {branch})...")
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", branch, clone_url, str(target)],
                check=True, capture_output=True, text=True,
            )
        _update_source_status(src_id, "cloned")
        log(f"  {name}: ready.")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        log(f"  ERROR cloning {name}: {err}")
        _update_source_status(src_id, "error", error=err)
        raise


def remove_source(source_id: str, delete_files: bool = True) -> None:
    """Remove source record from settings. Optionally delete cloned files."""
    from livedocs.settings import get_settings, save_settings

    s = get_settings()
    sources = [src for src in s.sources if src["id"] != source_id]

    if delete_files:
        removed = [src for src in s.sources if src["id"] == source_id]
        for src in removed:
            if src["kind"] in ("github", "upload"):
                target = _corpus_dir(src["corpus"]) / src["name"]
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)

    save_settings({"sources": sources, "deployed": False})


def _upsert_source(source: dict) -> None:
    from livedocs.settings import get_settings, save_settings
    s = get_settings()
    existing = [src for src in s.sources if src["id"] != source["id"]]
    save_settings({"sources": existing + [source]})


def _update_source_status(src_id: str, status: str, error: str = "") -> None:
    from livedocs.settings import get_settings, save_settings
    s = get_settings()
    sources = []
    for src in s.sources:
        if src["id"] == src_id:
            src = dict(src, status=status, error=error)
        sources.append(src)
    save_settings({"sources": sources})


def update_source_counts(src_id: str, counts: dict) -> None:
    from livedocs.settings import get_settings, save_settings
    s = get_settings()
    sources = []
    for src in s.sources:
        if src["id"] == src_id:
            src = dict(src, status="indexed", counts=counts)
        sources.append(src)
    save_settings({"sources": sources})
