"""Score one transfer from a clinical record and an embryo image."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch

from embryofusion.config import load_config, resolve
from embryofusion.data.processing import BINARY, CATEGORICAL, NUMERIC, load_image, normalise_image
from embryofusion.models.fusion import FusionNet


class FusionPredictor:
    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir) if model_dir else resolve(load_config().artifacts.model_dir)
        checkpoint = torch.load(model_dir / "fusion_model.pt", map_location="cpu", weights_only=True)
        self.image_size = checkpoint["image_size"]
        self.model = FusionNet(checkpoint["experiment"]["mode"], checkpoint["n_tabular"], self.image_size, checkpoint["model_cfg"],
                               checkpoint["experiment"].get("image_encoder", "cnn"))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.preprocessor = joblib.load(model_dir / "tabular_preprocessor.joblib")

    def predict(self, record: dict, image_path: str | Path | None = None) -> dict:
        """Return the fused probability and what each view contributes on its own.

        If no image is supplied the model falls back to the clinical record
        alone, which modality dropout during training makes possible.
        """
        frame = pd.DataFrame([record]).reindex(columns=NUMERIC + BINARY + CATEGORICAL)
        frame["frozen_transfer"] = frame["frozen_transfer"].fillna(0)
        frame["infertility_cause"] = frame["infertility_cause"].fillna("unexplained")
        tabular = torch.from_numpy(self.preprocessor.transform(frame).astype("float32"))
        if image_path is None:
            image, hide = torch.zeros(1, 1, self.image_size, self.image_size), "image"
        else:
            image, hide = torch.from_numpy(normalise_image(load_image(image_path, self.image_size)))[None, None], None
        with torch.no_grad():
            out = {"probability": float(torch.sigmoid(self.model(tabular, image, hide=hide)))}
            if hide is None:
                out["clinical_record_only"] = float(torch.sigmoid(self.model(tabular, image, hide="image")))
                out["image_only"] = float(torch.sigmoid(self.model(tabular, image, hide="tabular")))
        out["modalities_used"] = ["clinical_record"] + ([] if image_path is None else ["embryo_image"])
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
