"""Read files from a repository as it was at a given commit.

Repositories are cloned once as bare repos into ``data/repos`` and files are
read straight from git objects, so nothing is ever checked out. Blobs over
2 MB are left on the server (images, data files), which keeps clones small.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_DIR = Path(os.environ.get("ASTPRUNE_REPO_DIR", PROJECT_ROOT / "data" / "repos"))


class RepoError(RuntimeError):
    pass


def _git(args: list[str], cwd: Path | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=timeout)


def repo_path(repo: str) -> Path:
    return REPO_DIR / (repo.replace("/", "__") + ".git")


def ensure_repo(repo: str) -> Path:
    """Clone ``owner/name`` from GitHub if it is not in the cache yet."""
    path = repo_path(repo)
    if (path / "HEAD").exists():
        return path
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    if tmp.exists():
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
    proc = _git(["clone", "--bare", "--filter=blob:limit=2m", "--quiet",
                 f"https://github.com/{repo}.git", str(tmp)], timeout=7200)
    if proc.returncode != 0:
        raise RepoError(f"clone of {repo} failed: {proc.stderr.decode(errors='replace')[-400:]}")
    tmp.rename(path)
    return path


class Repo:
    """File access for one cached repository. Keeps a ``git cat-file`` process open."""

    def __init__(self, repo: str):
        self.name = repo
        self.path = ensure_repo(repo)
        self._proc: subprocess.Popen | None = None
        self._tree_cache: dict[str, dict[str, str]] = {}

    # -- commits ------------------------------------------------------------
    def has_commit(self, commit: str) -> bool:
        return _git(["cat-file", "-e", f"{commit}^{{commit}}"], cwd=self.path).returncode == 0

    def ensure_commit(self, commit: str) -> bool:
        """Make sure a commit is present, fetching it by SHA if the clone missed it."""
        if self.has_commit(commit):
            return True
        _git(["fetch", "--quiet", "--filter=blob:limit=2m", "origin", commit], cwd=self.path, timeout=1800)
        return self.has_commit(commit)

    # -- trees ----------------------------------------------------------------
    def tree(self, commit: str) -> dict[str, str]:
        """Map of ``path -> blob sha`` for every ``.py`` file at ``commit``."""
        if commit not in self._tree_cache:
            proc = _git(["ls-tree", "-r", "-z", commit], cwd=self.path)
            if proc.returncode != 0:
                raise RepoError(f"{self.name}@{commit}: {proc.stderr.decode(errors='replace')[-200:]}")
            tree: dict[str, str] = {}
            for entry in proc.stdout.split(b"\0"):
                if not entry:
                    continue
                meta, _, name = entry.partition(b"\t")
                parts = meta.split()
                if len(parts) == 3 and parts[1] == b"blob" and name.endswith(b".py"):
                    tree[name.decode("utf-8", errors="replace")] = parts[2].decode()
            if len(self._tree_cache) > 8:
                self._tree_cache.pop(next(iter(self._tree_cache)))
            self._tree_cache[commit] = tree
        return self._tree_cache[commit]

    # -- blobs ----------------------------------------------------------------
    def _batch(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                ["git", "cat-file", "--batch"], cwd=self.path,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        return self._proc

    def read_blob(self, sha: str) -> str | None:
        proc = self._batch()
        proc.stdin.write(sha.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) < 3 or header[1] != b"blob":
            return None
        size = int(header[2])
        data = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        return data.decode("utf-8", errors="replace")

    def read(self, commit: str, path: str) -> str | None:
        """Contents of ``path`` at ``commit``, or None if it does not exist."""
        sha = self.tree(commit).get(path)
        return self.read_blob(sha) if sha else None

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        self._proc = None

    def __enter__(self) -> "Repo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
