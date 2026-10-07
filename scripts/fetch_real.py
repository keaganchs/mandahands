"""Clone every real-hand source repo at its pinned commit into data/sources/<name> (sparse where possible).

  python scripts/fetch_real.py            # all repos
  python scripts/fetch_real.py talos nao  # some
"""

import subprocess
import sys
from pathlib import Path

from mandahands.embodiments.real.sources import REPOS

DEST = Path("data/sources")


def git(*args, cwd=None):
  subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch(name: str):
  url, commit, paths = REPOS[name]
  d = DEST / name
  if not d.exists():
    git("clone", "-q", "--filter=blob:none", "--no-checkout", url, str(d))
  if paths:
    git("sparse-checkout", "set", "--no-cone", *[f"/{p}" for p in paths], "/package.xml", "/LICENSE*", cwd=d)
  else:
    git("sparse-checkout", "disable", cwd=d)
  git("fetch", "-q", "--depth", "1", "origin", commit, cwd=d)
  git("checkout", "-q", commit, cwd=d)
  print(f"{name}: {commit}")


if __name__ == "__main__":
  for name in sys.argv[1:] or REPOS:
    fetch(name)
