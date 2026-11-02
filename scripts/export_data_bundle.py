"""Export the team's local corpus/index snapshot, excluding credentials and models."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("data/processed/passages.jsonl", "indexes/bm25/bm25.pkl",
            "indexes/dense/index.faiss", "indexes/dense/metadata.jsonl")
OPTIONAL = ("data/processed/corpus_stats.json", "data/raw/sec/manifest.jsonl")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/quantrag-data.zip")
    args = parser.parse_args()
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    if missing:
        parser.error("Missing required artifacts: " + ", ".join(missing))
    if args.output.exists():
        parser.error("Output exists; choose a new filename to preserve the existing bundle.")
    paths = [p for p in (*REQUIRED, *OPTIONAL) if (ROOT / p).is_file()]
    manifest = {p: {"size_bytes": (ROOT / p).stat().st_size, "sha256": digest(ROOT / p)} for p in paths}
    config_hash = digest(ROOT / "config.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for p in paths:
            bundle.write(ROOT / p, p)
        bundle.writestr("data-bundle-manifest.json", json.dumps({"files": manifest, "config_sha256": config_hash,
            "note": "Trusted team snapshot. BM25 is a pickle; do not load bundles from untrusted parties. This bundle contains no model weights or raw filing HTML."}, indent=2))
    if any(digest(ROOT / p) != manifest[p]["sha256"] for p in paths) or digest(ROOT / "config.json") != config_hash:
        raise RuntimeError("Inputs changed while exporting. Do not distribute this bundle; export again to a new filename.")
    with zipfile.ZipFile(args.output) as bundle:
        assert bundle.testzip() is None
    print(f"Created {args.output.resolve()} ({args.output.stat().st_size:,} bytes). Share privately with teammates; do not commit it.")


if __name__ == "__main__":
    main()
