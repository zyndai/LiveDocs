"""Local copy of the docs fetcher for ragpipline/. Clones or pulls the docs
repo into DOCS_DIR.

For PRIVATE repos: set GITHUB_TOKEN in .env. The token is injected into the
HTTPS URL at fetch time -- never stored in the cloned repo's .git/config, so
rotating the token doesn't break the existing clone.

For SSH-based access, set DOCS_REPO_URL in config.py to the SSH form
(e.g. "git@github.com:owner/repo.git"). The token is ignored for ssh URLs.
"""
import os
import re
import subprocess
from config import DOCS_REPO_URL, DOCS_BRANCH, DOCS_DIR


_TOKEN_RE = re.compile(r"(https://[^:/]+:)[^@]+(@)")


def _authed_url(url):
    """Inject GITHUB_TOKEN into an https:// URL if one is set. SSH URLs are returned as-is."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token or not url.startswith("https://"):
        return url
    # https://github.com/foo/bar -> https://oauth2:TOKEN@github.com/foo/bar
    return url.replace("https://", f"https://oauth2:{token}@", 1)


def _redact(cmd):
    """Replace any embedded token in logged commands so it doesn't end up in stdout."""
    return [_TOKEN_RE.sub(r"\1***\2", str(a)) for a in cmd]


def _run(cmd, cwd=None):
    print(f"$ {' '.join(_redact(cmd))}")
    subprocess.run(cmd, cwd=cwd, check=True)


def fetch():
    DOCS_DIR.parent.mkdir(parents=True, exist_ok=True)
    url = _authed_url(DOCS_REPO_URL)

    if (DOCS_DIR / ".git").exists():
        print(f"Updating existing clone at {DOCS_DIR}")
        # Pass the URL explicitly so the token isn't read from baked-in .git/config.
        _run(["git", "fetch", url, DOCS_BRANCH], cwd=DOCS_DIR)
        _run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=DOCS_DIR)
    else:
        if DOCS_DIR.exists():
            raise SystemExit(f"{DOCS_DIR} exists but is not a git repo. Delete it and retry.")
        print(f"Cloning {DOCS_REPO_URL} into {DOCS_DIR}")
        _run(["git", "clone", "--depth=1", "--branch", DOCS_BRANCH, url, str(DOCS_DIR)])

    print(f"Docs ready at {DOCS_DIR}")
    return DOCS_DIR


if __name__ == "__main__":
    fetch()
