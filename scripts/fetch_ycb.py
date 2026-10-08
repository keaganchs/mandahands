"""Download the YCB object models (google_16k meshes) into data/sources/ycb/<object>/.

  python scripts/fetch_ycb.py            # all objects in the official index that have google_16k models
  python scripts/fetch_ycb.py 025_mug    # some

License: CC-BY-4.0 (commercial use with attribution). Meshes stay local (data/ is gitignored); the object
library records each archive's sha256 so a later re-download can be checked against it.
"""

import hashlib
import io
import json
import sys
import tarfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://ycb-benchmarks.s3.amazonaws.com/data"
OUT = Path("data/sources/ycb")


def fetch(obj: str) -> tuple[str, str | None]:
  dest = OUT / obj
  if (dest / "sha256").exists():
    return obj, (dest / "sha256").read_text().strip()
  try:
    raw = urllib.request.urlopen(f"{BASE}/google/{obj}_google_16k.tgz", timeout=120).read()
  except urllib.error.HTTPError:
    return obj, None  # no google_16k model for this object
  dest.mkdir(parents=True, exist_ok=True)
  with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    tf.extractall(OUT, filter="data")
  digest = hashlib.sha256(raw).hexdigest()
  (dest / "sha256").write_text(digest)
  return obj, digest


def main():
  objs = sys.argv[1:] or json.load(urllib.request.urlopen(f"{BASE}/objects.json", timeout=60))["objects"]
  with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, objs))
  missing = [o for o, d in res if d is None]
  print(f"{len(res) - len(missing)} objects with google_16k models; none for: {', '.join(missing) or '-'}")


if __name__ == "__main__":
  main()
