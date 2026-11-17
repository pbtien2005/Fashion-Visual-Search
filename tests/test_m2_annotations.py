import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from evaluation.m2_annotations import load_manifest, import_m1_labels, apply_review


class AnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = import_m1_labels()

    def test_import_keeps_split_provenance_review_pending_and_known_overlap(self):
        manifest = json.loads(json.dumps(self.original))
        self.assertEqual(len(manifest["queries"]), 20)
        self.assertEqual(sum(row["previously_used_for_m2_development"] for row in manifest["queries"]), 15)
        self.assertTrue(all(row["label_status"] == "pending_human_review" for row in manifest["queries"]))
        self.assertTrue(all(row["source_url"] and row["author"] and row["sha256"] for row in manifest["queries"]))
        self.assertEqual(sum(row["mask_requested"] for row in manifest["queries"] if row["split"] == "validation"), 2)
        self.assertEqual(sum(row["mask_requested"] for row in manifest["queries"] if row["split"] == "test"), 3)
        self.assertFalse(manifest["independent_test"])

    def test_loader_rejects_hash_duplicates_invalid_ids_review_without_reviewer(self):
        source = json.loads(json.dumps(self.original))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            self.assertEqual(len(load_manifest(path)["queries"]), 20)
            for mutate in (
                lambda m: m["queries"].append(dict(m["queries"][0], query_id="different")),
                lambda m: m["queries"][0].update(sha256="0" * 64),
                lambda m: m["queries"][0].update(query_id="../escape"),
                lambda m: m["queries"][0].update(label_status="human_reviewed", reviewer=None),
                lambda m: m["queries"][0].update(class_name="unknown"),
                lambda m: m["queries"][0].update(image="../outside.png"),
            ):
                candidate = json.loads(json.dumps(source))
                mutate(candidate)
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_review_cannot_replace_images_or_claim_review_without_name(self):
        manifest = json.loads(json.dumps(self.original))
        review = json.loads(json.dumps(manifest))
        review["queries"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            apply_review(manifest, review, Path("tmp/unused"))
        review = json.loads(json.dumps(manifest))
        review["queries"][0].update(label_status="human_reviewed", reviewer="")
        with self.assertRaises(ValueError):
            apply_review(manifest, review, Path("tmp/unused"))

    def test_manual_polygon_export_import_writes_binary_canvas_mask(self):
        manifest = json.loads(json.dumps(self.original))
        review = json.loads(json.dumps(manifest))
        review["queries"][0].update(label_status="human_reviewed", reviewer="unit_test_fixture",
                                    mask_status="human_reviewed", mask_reviewer="unit_test_fixture",
                                    mask_polygons=[dict(points=[[30, 30], [80, 30], [80, 80], [30, 80]])])
        root = Path(__file__).resolve().parents[1]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            result = apply_review(manifest, review, Path(directory))
            mask = root / result["queries"][0]["mask_path"]
            with Image.open(mask) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.size, (manifest["queries"][0]["geometry_meta"]["width"], manifest["queries"][0]["geometry_meta"]["height"]))
                self.assertEqual(image.getpixel((40, 40)), 255)
                self.assertEqual(image.getpixel((0, 0)), 0)
            self.assertEqual(result["queries"][0]["mask_status"], "human_reviewed")
            with self.assertRaises(ValueError):
                apply_review(manifest, review, Path(directory))

    def test_invalid_review_with_polygons_does_not_leave_partial_mask(self):
        manifest = json.loads(json.dumps(self.original))
        review = json.loads(json.dumps(manifest))
        review["queries"][0].update(mask_status="human_reviewed", mask_reviewer="unit_test_fixture",
                                    mask_polygons=[dict(points=[[30, 30], [80, 30], [80, 80]])])
        root = Path(__file__).resolve().parents[1]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            with self.assertRaises(ValueError):
                apply_review(manifest, review, Path(directory))
            self.assertFalse(list(Path(directory).glob("*.png")))

    def test_review_bundle_contains_images_and_does_not_promote_pending_labels(self):
        from evaluation.prepare_m2_annotations import build_review_bundle
        manifest = json.loads(json.dumps(self.original))
        manifest["queries"] = manifest["queries"][:1]
        with tempfile.TemporaryDirectory() as directory:
            path = build_review_bundle(manifest, directory, contact_path=Path(directory) / "contact.jpg")
            page = path.read_text(encoding="utf-8")
            self.assertIn("data:image/jpeg;base64,", page)
            self.assertNotIn("__M2_REVIEW_PAYLOAD__", page)
            self.assertIn('"label_status": "pending_human_review"', page)
            self.assertIsNone(manifest["queries"][0]["reviewer"])
            self.assertTrue((Path(directory) / "contact.jpg").is_file())

    def test_assistant_mask_drafts_remain_pending_and_cannot_be_evaluation_references(self):
        from evaluation.m2_annotations import ROOT, read_json, reference_for
        from evaluation.prepare_m2_mask_drafts import add_drafts
        manifest = json.loads(json.dumps(self.original))
        with tempfile.TemporaryDirectory() as directory:
            result = add_drafts(manifest, read_json(ROOT / "evaluation/m2/mask_drafts.ai.json"), Path(directory) / "drafts.jpg")
        drafts = [row for row in result["queries"] if row.get("mask_polygons")]
        self.assertEqual(len(drafts), 5)
        for row in drafts:
            self.assertEqual(row["mask_status"], "pending_human_review")
            self.assertIsNone(row["mask_reviewer"])
            self.assertIsNone(row["mask_path"])
            self.assertIsNone(reference_for(row, row["geometry_meta"]))
        for field in ("preprocess", "image_hash"):
            spec = read_json(ROOT / "evaluation/m2/mask_drafts.ai.json")
            if field == "preprocess":
                spec["preprocess_config"]["max_side"] = 640
            else:
                spec["drafts"]["real_black_dress_02"]["image_sha256"] = "0" * 64
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                add_drafts(manifest, spec, Path(directory) / "should_not_exist.jpg")

    def test_prepare_cli_import_review_and_apply_pending_export(self):
        from copy import deepcopy
        from unittest.mock import patch
        from evaluation.prepare_m2_annotations import main
        source = deepcopy(self.original)
        source["queries"] = source["queries"][:1]
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root / "tmp") as directory:
            folder = Path(directory)
            manifest_path = folder / "pending.json"
            argv = ["prepare", "import-m1", "--manifest", str(manifest_path)]
            with patch("sys.argv", argv), patch("evaluation.prepare_m2_annotations.import_m1_labels", return_value=source):
                main()
                with self.assertRaises(SystemExit):
                    main()
            with patch("sys.argv", ["prepare", "review", "--manifest", str(manifest_path),
                                    "--output", str(folder / "page"), "--contact", str(folder / "contact.jpg")]):
                main()
            with patch("sys.argv", ["prepare", "apply-review", "--manifest", str(manifest_path),
                                    "--review", str(manifest_path), "--output", str(folder / "result.json"),
                                    "--mask-dir", str(folder / "masks")]):
                main()
            self.assertEqual(load_manifest(folder / "result.json")["queries"][0]["label_status"], "pending_human_review")


if __name__ == "__main__":
    unittest.main()
