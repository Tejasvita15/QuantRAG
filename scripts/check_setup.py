"""Report local readiness without printing keys or contacting providers."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("beautifulsoup4", "faiss-cpu", "numpy", "pandas", "python-dotenv",
            "rank-bm25", "requests", "sentence-transformers", "torch", "tqdm",
            "yfinance", "streamlit")
INDEXES = ("indexes/bm25/bm25.pkl", "indexes/dense/index.faiss", "indexes/dense/metadata.jsonl")
REQUIRED_BUNDLE_FILES = {"data/processed/passages.jsonl", *INDEXES}
ALLOWED_BUNDLE_FILES = REQUIRED_BUNDLE_FILES | {"data/processed/corpus_stats.json", "data/raw/sec/manifest.jsonl"}


def verify_bundle(root):
    """Verify approved bundle paths and checksums without deserializing indexes."""
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(block)
        return result.hexdigest()
    try:
        manifest = json.loads((root / "data-bundle-manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        if not isinstance(files, dict) or not REQUIRED_BUNDLE_FILES <= files.keys() or files.keys() - ALLOWED_BUNDLE_FILES:
            return {"verified": False, "error": "Bundle manifest has missing or unexpected file paths"}
        mismatches = [name for name, info in files.items()
                      if not (root / name).is_file() or (root / name).stat().st_size != info["size_bytes"]
                      or digest(root / name) != info["sha256"]]
        config_matches = digest(root / "config.json") == manifest["config_sha256"]
        return {"verified": not mismatches and config_matches, "mismatched_files": mismatches,
                "configuration_matches": config_matches}
    except (OSError, ValueError, TypeError, KeyError):
        return {"verified": False, "error": "Cannot validate bundle; check manifest, artifacts and config.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-bundle", action="store_true", help="Also verify the extracted data bundle checksums")
    args = parser.parse_args()
    versions, missing = {}, []
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    configured = {"FRED": False, "LLM": False}
    if "python-dotenv" not in missing:
        from dotenv import dotenv_values
        import os
        values = {**dotenv_values(ROOT / ".env"), **os.environ}
        configured = {"FRED": bool(values.get("FRED_API_KEY")),
                      "LLM": bool(values.get("LLM_MODEL") and (values.get("LLM_API_KEY") or values.get("OPENAI_API_KEY")))}
    supported = sys.version_info[:2] in {(3, 10), (3, 11)}
    bundle = verify_bundle(ROOT) if args.verify_bundle else None
    print(json.dumps({"python": sys.version.split()[0], "supported_python": supported,
                      "installed_packages": versions, "missing_packages": missing,
                      "env_file_present": (ROOT / ".env").is_file(),
                      "index_files": {p: (ROOT / p).is_file() for p in INDEXES},
                      "provider_configuration_present": configured,
                      "bundle_verification": bundle,
                      "note": "Presence checks only; no provider call, pickle load or index compatibility check was performed."}, indent=2))
    return 0 if supported and not missing and (bundle is None or bundle["verified"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
