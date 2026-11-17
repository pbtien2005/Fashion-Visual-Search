import csv
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from evaluation.benchmark.catalog import Catalog
from evaluation.benchmark.build_perturbed import CONDITIONS, SOURCE_MANIFESTS, build
from evaluation.benchmark.grade import load_judgments, merge_grades
from evaluation.benchmark.http_adapter import HttpSystem
from evaluation.benchmark.manifest import FIELDS, ROOT, load_manifest, portable_path, resolve_path, validate_splits
from evaluation.benchmark.pipeline import InProcessSystem, default_configs, temporary_env
from evaluation.benchmark.run import execute, make_parser, read_partial, verify_test_lock
from evaluation.benchmark.score import ratio, select_c3


class ScoreTests(unittest.TestCase):
    def test_choose_c3_on_validation_and_keep_baseline_on_tie(self):
        reports = {name: {"mrr_at_5": (0.5, 2)} for name in ("B0_current", "C3_w1", "C3_w2")}
        self.assertEqual(select_c3(reports), "B0_current")
        reports["C3_w2"]["mrr_at_5"] = (0.75, 2)
        self.assertEqual(select_c3(reports), "C3_w2")

    def test_false_acceptance_suppresses_small_rate(self):
        self.assertEqual(ratio((1, 4), 5), "1 ca / 4")
        self.assertEqual(ratio((1, 5), 5), "0.200 (1/5)")


class RunLockTests(unittest.TestCase):
    def test_duplicate_run_refused_before_loading_files(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "test").mkdir()
            args = make_parser().parse_args(["test", "--run-dir", folder])
            with self.assertRaises(FileExistsError):
                execute(args)

    def test_assets_options_and_seen_image(self):
        snapshot = {"model_sha256": "model", "catalog_sha256": "catalog", "conf": .5, "crop_config": {}}
        lock = {**snapshot, "validation_images": ["val"], "configs": {"B0_current": {"kind": "rerank"}}}
        definitions = lock["configs"]
        verify_test_lock(lock, snapshot, [{"sha256": "test"}], definitions)
        for field, change in [("model_sha256", "other"), ("catalog_sha256", "other"),
                              ("conf", .6), ("crop_config", {"expand_ratio": .3})]:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                verify_test_lock(lock, snapshot | {field: change}, [{"sha256": "test"}], definitions)
        with self.assertRaisesRegex(ValueError, "locked validation"):
            verify_test_lock(lock, snapshot, [{"sha256": "val", "query_id": "leak"}], definitions)


class RunPersistenceTests(unittest.TestCase):
    """Drive execute() with a fake system: failures, interruption and resume."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("labels.csv", "model.pt", "catalog.json"):
            (self.root / name).write_bytes(name.encode())
        self.labels = []
        for query_id in ("q1", "q2", "q3"):
            path = self.root / f"{query_id}.jpg"
            path.write_bytes(query_id.encode())
            self.labels.append({"query_id": query_id, "product_id": "p", "condition": "original",
                                "path": path, "sha256": hashlib.sha256(query_id.encode()).hexdigest()})
        self.calls = []

    def run_stage(self, run_dir, behavior, resume=False):
        def run_query(pixels, config):
            query_id = pixels.decode()
            self.calls.append((query_id, config.name))
            outcome = behavior.get((query_id, config.name))
            if outcome is not None:
                raise outcome
            return {"geometry": {"width": 1, "height": 1}, "quality": {"gamma_applied": False},
                    "detections": [], "timings_ms": {}, "detection_measured": True}

        system = Mock(crop_config={}, run_query=run_query, warm_text_cache=Mock(return_value=1.0))
        catalog = Mock(sha256=hashlib.sha256(b"catalog.json").hexdigest())
        argv = ["robustness", "--run-dir", str(run_dir), "--labels", str(self.root / "labels.csv"),
                "--model", str(self.root / "model.pt"), "--catalog", str(self.root / "catalog.json")]
        with patch("evaluation.benchmark.run.Catalog.load", return_value=catalog), \
                patch("evaluation.benchmark.run.InProcessSystem", return_value=system), \
                patch("evaluation.benchmark.build_perturbed.load_perturbed", return_value=self.labels), \
                patch("evaluation.benchmark.build_perturbed.write_robustness_report"), \
                patch("builtins.print"):
            execute(make_parser().parse_args(argv + (["--resume"] if resume else [])))

    def test_failed_query_is_kept_as_a_miss(self):
        run_dir = self.root / "run"
        self.run_stage(run_dir, {("q2", "B0_current"): RuntimeError("boom")})
        saved = json.loads((run_dir / "robustness/results.json").read_text(encoding="utf-8"))
        self.assertEqual([query["query_id"] for query in saved["queries"]], ["q1", "q2", "q3"])
        failed = saved["queries"][1]["results"]["B0_current"]
        self.assertIn("boom", failed["error"])
        self.assertEqual(failed["detections"], [])
        self.assertEqual(len(saved["errors"]), 1)
        self.assertFalse((run_dir / "robustness/queries.partial.jsonl").exists())

    def test_interrupted_run_resumes_without_repeating_queries(self):
        run_dir = self.root / "run"
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage(run_dir, {("q2", "A0_whole_image"): KeyboardInterrupt()})
        self.assertFalse((run_dir / "robustness/results.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_stage(run_dir, {})
        self.calls.clear()
        self.run_stage(run_dir, {}, resume=True)
        self.assertEqual({query_id for query_id, _ in self.calls}, {"q2", "q3"})
        saved = json.loads((run_dir / "robustness/results.json").read_text(encoding="utf-8"))
        self.assertEqual([query["query_id"] for query in saved["queries"]], ["q1", "q2", "q3"])
        self.assertEqual(saved["resumed_after"], 1)

    def test_setup_failure_releases_the_reservation(self):
        run_dir = self.root / "run"
        with self.assertRaises(RuntimeError):
            self.run_stage(run_dir, {("q1", "A0_whole_image"): RuntimeError("no model")})
        self.assertFalse((run_dir / "robustness").exists())

    def test_resume_refuses_another_configuration(self):
        partial = self.root / "queries.partial.jsonl"
        partial.write_text(json.dumps({"identity": {"conf": 0.5}}) + "\n" + '{"query_id": "q1"}\n{"query_id"',
                           encoding="utf-8")
        self.assertEqual(read_partial(partial, {"conf": 0.5}), [{"query_id": "q1"}])
        with self.assertRaisesRegex(ValueError, "different configuration"):
            read_partial(partial, {"conf": 0.4})


class PortablePathTests(unittest.TestCase):
    def test_repo_paths_are_stored_relative(self):
        inside = ROOT / "evaluation" / "bench_test.csv"
        self.assertEqual(portable_path(inside), "evaluation/bench_test.csv")
        self.assertEqual(resolve_path(portable_path(inside)), inside)
        outside = Path(tempfile.gettempdir()).resolve() / "elsewhere.csv"
        self.assertEqual(resolve_path(portable_path(outside)), outside)


class GradeTests(unittest.TestCase):
    def test_merge_overwrites_and_rejects_invalid_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            grades = {
                "grader": "tester", "exported_at": "2026-11-16T00:00:00Z",
                "pairs": [{"query_id": "a", "product_id": "p", "type_ok": 1, "color_ok": 0, "acceptable": 1}],
                "queries": [{"query_id": "a", "no_match": False, "expanded_reviewed": False},
                            {"query_id": "b", "no_match": True, "expanded_reviewed": True}],
            }
            self.assertEqual(merge_grades(grades, root)[:2], (1, 0))
            pairs, status = load_judgments(root)
            self.assertEqual(pairs["a", "p"]["grader"], "tester")
            self.assertEqual(status, {"a": "has_match", "b": "no_match"})
            grades["pairs"][0]["type_ok"] = 0
            with self.assertRaisesRegex(ValueError, "acceptable"):
                merge_grades(grades, root)
            self.assertEqual(load_judgments(root), (pairs, status))
            grades["pairs"][0]["acceptable"] = 0
            self.assertEqual(merge_grades(grades, root)[:2], (0, 1))
            self.assertEqual(load_judgments(root)[1]["a"], "unconfirmed")


class PerturbationTests(unittest.TestCase):
    def test_rebuild_has_identical_hashes(self):
        from evaluation.run_m1_ablation import file_hash

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (48, 36), (140, 80, 20)).save(root / "base.jpg")
            source = [{"image": "base.jpg", "expected_items": [{"acceptable_product_ids": ["p"]}]}]
            for filename in SOURCE_MANIFESTS:
                (root / filename).write_text(json.dumps(source))
            rows = build(root)
            self.assertEqual({row["condition"] for row in rows}, set(CONDITIONS))
            before = [file_hash(root / row["image"]) for row in rows]
            with self.assertRaises(FileExistsError):
                build(root)
            self.assertEqual(rows, build(root, force=True))
            self.assertEqual(before, [file_hash(root / row["image"]) for row in rows])


class HttpTests(unittest.TestCase):
    def test_maps_both_rank_lists(self):
        base = {"class_name": "short_sleeve_top", "confidence": .9, "bbox": [0, 0, 40, 40]}
        response = {"success": True, "detected_items": [
            base | {"is_confident_match": True, "similar_products": [{"id": "1"}]},
            base | {"is_confident_match": False, "low_confidence_products": [{"id": "2"}],
                    "reject_reason": "final_score_too_low"},
        ]}
        buf = io.BytesIO()
        Image.new("RGB", (40, 40), "red").save(buf, format="PNG")
        with patch("evaluation.benchmark.http_adapter.post_query_image", return_value=(response, 70)) as post:
            result = HttpSystem("http://localhost:8000").run_query(buf.getvalue(), "input.png")
        self.assertEqual(post.call_args.args[3], 10)
        self.assertEqual(result["geometry"]["width"], 40)
        first, second = result["detections"]
        self.assertEqual([first["ranked"][0]["id"], second["ranked"][0]["id"]], ["1", "2"])
        self.assertTrue(first["accepted"])
        self.assertFalse(second["accepted"])
        self.assertFalse(result["candidates_available"])
        self.assertEqual(result["timings_ms"]["total"], 70)
        self.assertIsNone(result["timings_ms"]["detect"])


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for crop JPEG round-trip")
class PipelineTests(unittest.TestCase):
    def test_shared_detect_and_embedding_only_order(self):
        crop = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
        detect = Mock(return_value=[{
            "class_name": "short_sleeve_top", "bbox": [0, 0, 64, 64], "confidence": .9,
            "variants": [{"name": "tight", "image": crop}],
        }])
        embed = Mock(return_value=[1, 0])
        catalog = Catalog([
            {"id": "b", "embedding": [-1, 0], "title": "Red shirt"},
            {"id": "a", "embedding": [-1, 1], "title": "Blue trousers"},
        ], dimension=2)
        system = InProcessSystem(catalog, "unused.pt", detect_fn=detect, embed_fn=embed)
        configs = default_configs()
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
        with patch("api.search_service.compute_text_style_scores", return_value={}):
            current = system.run_query(buf.getvalue(), configs["B0_current"])
            embedding = system.run_query(buf.getvalue(), configs["C2_embedding_only"])
        detect.assert_called_once()
        embed.assert_called_once()
        self.assertEqual(embedding["geometry"]["width"], 64)
        self.assertEqual(embedding["detections"][0]["query_color_name"], "red")
        self.assertEqual([row["id"] for row in embedding["detections"][0]["ranked"]], ["a", "b"])
        self.assertEqual(embedding["detections"][0]["candidate_ids"], ["a", "b"])
        self.assertEqual(current["timings_ms"]["retrieve"], embedding["timings_ms"]["retrieve"])
        self.assertEqual(configs["C2_embedding_only"].kind, "embedding")

    def test_crop_without_embedding_scores_as_empty_result(self):
        crop = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
        detect = Mock(return_value=[{
            "class_name": "short_sleeve_top", "bbox": [0, 0, 64, 64], "confidence": .9,
            "variants": [{"name": "tight", "image": crop}],
        }])
        catalog = Catalog([{"id": "a", "embedding": [1, 0], "title": "Red shirt"}], dimension=2)
        system = InProcessSystem(catalog, "unused.pt", detect_fn=detect, embed_fn=Mock(return_value=None))
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
        configs = default_configs()
        for name in ("B0_current", "C2_embedding_only", "A0_whole_image"):
            with self.subTest(name=name):
                detection = system.run_query(buf.getvalue(), configs[name])["detections"][0]
                self.assertEqual(detection["ranked"], [])
                self.assertFalse(detection["accepted"])
                self.assertIn("error", detection)


class EnvironmentTests(unittest.TestCase):
    def test_restores_after_exception(self):
        with patch.dict(os.environ, {"BENCH_PRESENT": "old"}, clear=True):
            with self.assertRaises(RuntimeError):
                with temporary_env({"BENCH_PRESENT": "new", "BENCH_NEW": "set"}):
                    self.assertEqual(os.environ["BENCH_PRESENT"], "new")
                    raise RuntimeError()
            self.assertEqual(os.environ["BENCH_PRESENT"], "old")
            self.assertNotIn("BENCH_NEW", os.environ)


class CatalogTests(unittest.TestCase):
    def test_cosine_order_ties_and_limit(self):
        rows = [
            {"id": product_id, "embedding": vector}
            for product_id, vector in [("c", [0, 2]), ("b", [3, 0]), ("a", [2, 0])]
        ]
        rows.extend({"id": f"d{i:03}", "embedding": [-1, 0]} for i in range(100))
        catalog = Catalog(rows, dimension=2)
        ranked = catalog.search([4, 0], 1000)
        self.assertEqual(len(ranked), 100)
        self.assertEqual([row["id"] for row in ranked[:3]], ["a", "b", "c"])
        self.assertEqual([row["similarity"] for row in ranked[:3]], [1, 1, 0])
        self.assertEqual(len(catalog.search([1, 0], 0)), 1)
        self.assertNotIn("embedding", catalog.get("a"))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def make_row(self, name, lighting, **changes):
        (self.root / f"{name}.jpg").write_bytes(name.encode())
        return {
            "query_id": name, "image": f"{name}.jpg", "lighting_group": lighting,
            "class_name": "short_sleeve_top", "color": "red", "bbox": "0 0 1 1",
            "source": "fixture", "license": "fixture", "author": name, "license_url": "",
        } | changes

    def write_manifest(self, name, rows):
        path = self.root / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_invalid_labels_and_duplicate_images(self):
        normal = self.make_row("normal", "normal")
        for changes in [{"query_id": "normal"}, {"image": "normal.jpg"},
                        {"bbox": "0 0 1.2 1"}, {"color": "invented"}]:
            with self.subTest(changes=changes):
                path = self.write_manifest("bad", [normal, self.make_row("dark", "dark", **changes)])
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_cross_split_duplicate(self):
        val = self.write_manifest("val", [self.make_row("a", "normal"), self.make_row("b", "dark")])
        test = self.write_manifest("test", [self.make_row("c", "normal", image="a.jpg"),
                                           self.make_row("d", "dark")])
        with self.assertRaisesRegex(ValueError, "image"):
            validate_splits(load_manifest(val), load_manifest(test))

