"""Tests for the artifact store: sidecar creation, manifest, frames, detections, checkpoints."""

import tempfile
from pathlib import Path

from analysis.store import ArtifactStore


class TestArtifactStore:
    def test_create_sidecar_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc123", "cfg456")
            store.create()
            run_dir = store.run_dir
            assert run_dir.exists()
            assert (run_dir / "manifest.json").exists()
            assert (run_dir / "frames").is_dir()
            assert (run_dir / "detections").is_dir()
            assert (run_dir / "checkpoints").is_dir()

    def test_manifest_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc123", "cfg456")
            store.create()
            m = store._manifest
            assert m["sidecar_version"] == 1
            assert m["video_hash"] == "abc123"
            assert m["config_hash"] == "cfg456"
            assert m["run_id"] == store.run_id

    def test_record_pass_start_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            store.record_pass_start("p1", {"model": "test.pt"})
            passes = store._manifest["passes"]
            assert "p1" in passes
            assert passes["p1"]["status"] == "running"
            store.record_pass_complete("p1", {"frames": 10})
            assert passes["p1"]["status"] == "complete"
            assert passes["p1"]["stats"]["frames"] == 10

    def test_record_pass_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            store.record_pass_start("p1", {})
            store.record_pass_error("p1", "out of memory")
            assert store._manifest["passes"]["p1"]["status"] == "error"

    def test_add_frame_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            store.add_frame("p1", 0, {"pts_ms": 0.0, "stab_offset": [0.0, 0.0]})
            store.add_frame("p1", 1, {"pts_ms": 40.0, "stab_offset": [1.0, 0.0]})
            fpath = store.run_dir / "frames" / "p1.jsonl"
            lines = fpath.read_text().strip().split("\n")
            assert len(lines) == 2
            import json

            assert json.loads(lines[0])["frame_no"] == 0
            assert json.loads(lines[1])["frame_no"] == 1

    def test_add_detection_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            store.add_detection(
                "p1",
                0,
                0,
                {
                    "xyxy_raw": [10, 20, 50, 80],
                    "conf": 0.9,
                    "cls": "person",
                    "embedding": [0.1, 0.2, 0.3],
                },
            )
            fpath = store.run_dir / "detections" / "p1.jsonl"
            lines = fpath.read_text().strip().split("\n")
            assert len(lines) == 1
            import json

            d = json.loads(lines[0])
            assert d["det_id"] == 0
            assert d["frame_no"] == 0
            assert d["embedding"] == [0.1, 0.2, 0.3]

    def test_checkpoint_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            state = {"last_frame": 42, "det_id": 100, "processed": 43}
            cp_path = store.save_checkpoint("p1", state)
            assert Path(cp_path).exists()
            loaded = store.load_checkpoint("p1")
            assert loaded == state

    def test_load_checkpoint_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            assert store.load_checkpoint("nonexistent") is None

    def test_get_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(tmp, "abc", "cfg")
            store.create()
            assert store.get_last_frame("p1") == -1  # no file yet
            store.add_detection(
                "p1",
                0,
                0,
                {
                    "xyxy_raw": [0, 0, 1, 1],
                    "conf": 0.5,
                    "cls": "person",
                    "embedding": None,
                },
            )
            store.add_detection(
                "p1",
                5,
                1,
                {
                    "xyxy_raw": [0, 0, 1, 1],
                    "conf": 0.5,
                    "cls": "person",
                    "embedding": None,
                },
            )
            assert store.get_last_frame("p1") == 5

    def test_config_hash_deterministic(self):
        s1 = {"model": "a.pt", "imgsz": 640, "conf": 0.3}
        s2 = {"model": "a.pt", "imgsz": 640, "conf": 0.3}
        h1 = ArtifactStore.config_hash_from_settings(s1)
        h2 = ArtifactStore.config_hash_from_settings(s2)
        assert h1 == h2
        assert len(h1) == 16

    def test_config_hash_different_configs(self):
        s1 = {"model": "a.pt", "imgsz": 640}
        s2 = {"model": "a.pt", "imgsz": 1280}
        h1 = ArtifactStore.config_hash_from_settings(s1)
        h2 = ArtifactStore.config_hash_from_settings(s2)
        assert h1 != h2
