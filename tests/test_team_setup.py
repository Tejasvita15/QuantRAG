import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_setup import REQUIRED_BUNDLE_FILES, verify_bundle


class BundleVerificationTests(unittest.TestCase):
    def make_bundle(self, root):
        manifest = {"files": {}}
        for name in REQUIRED_BUNDLE_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test artifact; not a real pickle or index")
            manifest["files"][name] = {"size_bytes":path.stat().st_size,
                                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        (root / "config.json").write_text("{}", encoding="utf-8")
        manifest["config_sha256"] = hashlib.sha256(b"{}").hexdigest()
        (root / "data-bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_matching_bundle_verifies_without_loading_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_bundle(root)
            self.assertTrue(verify_bundle(root)["verified"])

    def test_changed_index_or_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_bundle(root)
            (root / "indexes/dense/index.faiss").write_bytes(b"changed")
            self.assertFalse(verify_bundle(root)["verified"])
            self.make_bundle(root)
            (root / "config.json").write_text('{"different":true}', encoding="utf-8")
            self.assertFalse(verify_bundle(root)["verified"])

    def test_unexpected_paths_and_missing_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(verify_bundle(root)["verified"])
            manifest = self.make_bundle(root)
            manifest["files"]["../outside"] = {}
            (root / "data-bundle-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertFalse(verify_bundle(root)["verified"])


if __name__ == "__main__":
    unittest.main()
