import json

import pytest
import torch

from embryofusion.config import Config, load_config
from embryofusion.models.fusion import FusionNet

MODEL_CFG = {"tabular_hidden": [16, 8], "image_embedding": 16, "fusion_hidden": 8, "dropout": 0.0, "modality_dropout": 0.5,
             "vit": {"pretrained_name": None, "patch_size": 8, "hidden_size": 32, "layers": 1, "heads": 2}}


@pytest.mark.parametrize("mode,encoder", [("tabular", "cnn"), ("image", "cnn"), ("fusion", "cnn"), ("fusion", "vit")])
def test_every_variant_returns_one_logit_per_case(mode, encoder):
    model = FusionNet(mode, 12, 32, MODEL_CFG, encoder).eval()
    assert model(torch.randn(4, 12), torch.randn(4, 1, 32, 32)).shape == (4,)


def test_hiding_a_branch_changes_the_fused_output_only_in_fusion_mode():
    torch.manual_seed(0)
    model = FusionNet("fusion", 12, 32, MODEL_CFG).eval()
    tab, img = torch.randn(4, 12), torch.randn(4, 1, 32, 32)
    full = model(tab, img)
    assert not torch.allclose(full, model(tab, img, hide="image")) and not torch.allclose(full, model(tab, img, hide="tabular"))
    assert torch.allclose(model(tab, img, hide="image"), model(tab, torch.randn(4, 1, 32, 32), hide="image"))    # image truly ignored


def test_modality_dropout_is_inactive_at_inference():
    torch.manual_seed(0)
    model = FusionNet("fusion", 12, 32, MODEL_CFG).eval()
    tab, img = torch.randn(8, 12), torch.randn(8, 1, 32, 32)
    assert torch.allclose(model(tab, img), model(tab, img))


def test_unknown_mode_and_encoder_are_rejected():
    with pytest.raises(ValueError):
        FusionNet("audio", 12, 32, MODEL_CFG)
    with pytest.raises(ValueError):
        FusionNet("fusion", 12, 32, MODEL_CFG, "resnet9000")


@pytest.mark.slow
def test_pipeline_end_to_end_and_prediction(tiny_root, tmp_path):
    from embryofusion.inference.predict import FusionPredictor
    from embryofusion.training.pipeline import run_pipeline

    cfg = dict(load_config())
    cfg["data"] = {**cfg["data"], "root": str(tiny_root), "image_size": 32}
    cfg["model"] = MODEL_CFG
    cfg["training"] = {**cfg["training"], "epochs": 1, "batch_size": 32}
    cfg["tracking"] = {"project": "test", "mode": "disabled"}
    cfg["evaluation"] = {"bootstrap_rounds": 20}
    cfg["artifacts"] = {"model_dir": str(tmp_path / "models"), "report_dir": str(tmp_path / "reports"), "figure_dir": str(tmp_path / "figs")}
    report = run_pipeline(Config(cfg))
    assert {"tabular_only", "image_only_cnn", "fusion_cnn", "fusion_vit", "late_fusion"} <= set(report["results"])
    assert set(report["modality_ablation"]) == {"full", "image_hidden", "tabular_hidden", "image_shuffled", "tabular_shuffled"}
    assert json.loads((tmp_path / "reports" / "tracking" / "run_fusion_cnn.json").read_text())["history"]

    predictor = FusionPredictor(tmp_path / "models")
    record = {"female_age": 33, "bmi": 23.0, "amh_ng_ml": 2.5, "endometrial_thickness_mm": 10.0, "previous_failed_transfers": 0,
              "infertility_duration_years": 2.0, "infertility_cause": "unexplained", "frozen_transfer": 1}
    image = next((tiny_root / "images").glob("*.png"))
    both = predictor.predict(record, image)
    assert 0 < both["probability"] < 1 and both["modalities_used"] == ["clinical_record", "embryo_image"]
    record_only = predictor.predict({"female_age": 40})
    assert 0 < record_only["probability"] < 1 and record_only["modalities_used"] == ["clinical_record"]
