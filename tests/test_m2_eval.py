from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from evaluation.run_m2_eval import evaluate, save_report
from object_detection.detector import ClothingDetector
from object_detection.image_preprocessing import PreprocessConfig


class M2EvalTests(unittest.TestCase):
    def test_cli_writes_traceable_report_and_refuses_existing_output_or_empty_split(self):
        from evaluation.run_m2_eval import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (80, 80), "red").save(image)
            row = dict(query_id="cli_fixture", target_id="cli_fixture_target", image=str(image),
                       sha256=hashlib.sha256(image.read_bytes()).hexdigest(), split="validation",
                       label_status="pending_human_review", mask_status="not_annotated", class_name="short_sleeve_top",
                       bbox_normalized=[0, 0, 1, 1], source_url="synthetic_fixture")
            # The CLI validator is mocked only to use a temporary absolute fixture image.
            # Production manifests still require relative workspace paths.
            row["image"] = image.name
            manifest = dict(dataset_kind="synthetic_fixture_only", preprocess_config=asdict(PreprocessConfig()), queries=[row])
            manifest_path, model_path = root / "manifest.json", root / "model.pt"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model_path.write_bytes(b"synthetic_model_fixture_not_real_weights")
            detector = ClothingDetector()
            box = SimpleNamespace(xyxy=[[0, 0, 80, 80]], cls=[0], conf=[.9])
            detector._model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
            argv = ["eval", "--manifest", str(manifest_path), "--model", str(model_path),
                    "--output", str(root / "run"), "--modes", "off", "raw", "--split", "validation"]
            from evaluation.run_m2_eval import evaluate as real_evaluate
            def fixture_eval(m, d, **kwargs):
                return real_evaluate(m, d, root=root, **kwargs)
            with patch("sys.argv", argv), patch("evaluation.run_m2_eval.load_manifest", side_effect=lambda _: deepcopy(manifest)), patch(
                "evaluation.run_m2_eval.ClothingDetector", return_value=detector
            ), patch("evaluation.run_m2_eval.evaluate", side_effect=fixture_eval):
                self.assertEqual(main(), 0)
                with self.assertRaises(SystemExit):
                    main()
            result = json.loads((root / "run/results.json").read_text())
            self.assertEqual(result["summary"]["off"]["reviewed"]["target_count"], 0)
            self.assertTrue(result["model_sha256"])
            argv[argv.index("--output") + 1] = str(root / "empty")
            argv[-1] = "test"
            with patch("sys.argv", argv), patch("evaluation.run_m2_eval.load_manifest", return_value=deepcopy(manifest)):
                with self.assertRaises(SystemExit):
                    main()

    def test_one_detection_pass_modes_same_bbox_transport_and_reference_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.full((100, 100, 3), 20, np.uint8)
            source[25:75, 25:75] = (180, 80, 40)
            Image.fromarray(source).save(root / "image.png")
            reference = np.zeros((100, 100), np.uint8)
            reference[25:75, 25:75] = 255
            Image.fromarray(reference).save(root / "mask.png")
            manifest = {"dataset_kind": "synthetic_fixture_only", "preprocess_config": asdict(PreprocessConfig()),
                        "queries": [dict(query_id="synthetic", target_id="synthetic_target", image="image.png",
                                         sha256=hashlib.sha256((root / "image.png").read_bytes()).hexdigest(),
                                         split="validation", label_status="human_reviewed", mask_status="human_reviewed",
                                         mask_path="mask.png", mask_coordinate_space="m1_working_image",
                                         bbox_normalized=[0, 0, 1, 1], class_name="short_sleeve_top", source_url="fixture")]}
            box = SimpleNamespace(xyxy=[[0, 0, 100, 100]], cls=[0], conf=[.9])
            detector = ClothingDetector()
            detector._model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=[box])]))
            report = evaluate(manifest, detector, root=root, warmup=False, output=root)
            detector._model.predict.assert_called_once()
            self.assertEqual([row.get("error") for row in report["rows"]], [None] * 3)
            self.assertEqual(report["summary"]["off"]["reviewed"]["mask_iou_end_to_end_mean"], 0)
            raw = report["summary"]["raw"]["reviewed"]
            self.assertEqual(raw["mask_iou_conditional_mean"], 1)
            self.assertEqual(raw["mask_iou_end_to_end_mean"], 1)
            self.assertEqual(report["rows"][1]["HSV_valid_pixels"], 2500)
            save_report(report, root)
            self.assertTrue((root / "targets.csv").is_file())

    def test_inference_error_keeps_reviewed_targets_and_reference_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (80, 80)).save(root / "image.png")
            Image.new("L", (80, 80), 255).save(root / "mask.png")
            row = dict(query_id="failure", target_id="failure_target", image="image.png", sha256="fixture",
                       split="test", label_status="human_reviewed", mask_status="human_reviewed",
                       mask_path="mask.png", mask_coordinate_space="m1_working_image",
                       class_name="short_sleeve_top", bbox_normalized=[0, 0, 1, 1], source_url="fixture")
            manifest = dict(dataset_kind="synthetic_fixture_only", preprocess_config=asdict(PreprocessConfig()), queries=[row])
            detector = ClothingDetector()
            detector._model = SimpleNamespace(predict=Mock(side_effect=RuntimeError("fixture failure")))
            report = evaluate(manifest, detector, root=root, warmup=False)
            for summary in report["summary"].values():
                self.assertEqual(summary["reviewed"]["target_count"], 1)
                self.assertEqual(summary["reviewed"]["target_detection_recall"], 0)
                self.assertEqual(summary["reviewed"]["errors"], 1)
                self.assertEqual(summary["reviewed"]["mask_iou_end_to_end_mean"], 0)


if __name__ == "__main__":
    unittest.main()
