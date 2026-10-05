"""Paired multimodal dataset generator.

Each simulated embryo transfer has two views of the same case:

* a clinical record for the patient (age, BMI, AMH, endometrial thickness,
  history, cause of infertility, fresh or frozen transfer), and
* a day 5 image of the transferred blastocyst.

Embryo quality is never written to the clinical table. It is only visible in
the image, through the size and compactness of the inner cell mass, the
cohesion of the trophectoderm and the degree of expansion. Clinical pregnancy
depends on both the patient and the embryo, with an interaction: a good embryo
helps more in a receptive uterus. The two views are also correlated (older
patients tend to have poorer embryos), which is what makes the comparison of
single modality and fused models realistic.

The generator deliberately leaves a few orphan records, duplicate rows and
missing values in the raw files so that the data processing code has real work
to do.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from embryofusion.data.render import render_blastocyst

CAUSES = ["unexplained", "male_factor", "tubal", "ovulatory", "endometriosis"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_paired_dataset(root: str | Path, n_transfers: int, render_size: int = 192, image_size: int = 80, seed: int = 5) -> dict:
    """Write ``clinical.csv``, ``image_manifest.csv`` and ``images/`` under ``root``."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    n = n_transfers
    age = np.clip(rng.normal(35, 4.5, n), 22, 46).round(1)
    bmi = np.clip(21 + rng.gamma(3.0, 1.6, n), 17, 45).round(1)
    amh = np.clip(np.exp(1.2 - 0.085 * (age - 30) + rng.normal(0, 0.6, n)), 0.05, 20).round(2)
    endometrium = np.clip(rng.normal(9.5, 1.8, n), 5, 16).round(1)
    failed = rng.poisson(0.6, n).clip(0, 5)
    duration = np.clip(0.5 + rng.gamma(2.2, 1.3, n), 0.5, 15).round(1)
    cause = rng.choice(CAUSES, n, p=[0.32, 0.28, 0.14, 0.16, 0.10])
    frozen = (rng.random(n) < 0.55).astype(int)

    quality = rng.normal(-0.07 * (age - 35), 1.0)
    icm = np.digitize(quality + rng.normal(0, 0.7, n), [-0.35, 0.95])
    icm = 2 - icm                                            # 0 is grade A (best), 2 is grade C
    te = 2 - np.digitize(quality + rng.normal(0, 0.7, n), [-0.5, 0.8])
    expansion = np.clip(np.round(3.2 + 0.5 * quality + rng.normal(0, 0.8, n)), 2, 5).astype(int)   # index 2 to 5 is Gardner 3 to 6

    embryo = 0.6 * (1 - icm) + 0.5 * (1 - te) + 0.2 * (expansion - 3)
    uterus = (-0.07 * (age - 35) + 0.22 * np.clip(endometrium - 9.5, -4, 2.5) - 0.8 * (endometrium < 7)
              - 0.03 * np.clip(bmi - 24, 0, None) - 0.2 * failed + 0.15 * frozen - 0.15 * (cause == "endometriosis"))
    logit = -0.75 + uterus + embryo * (1 + 0.25 * np.tanh(uterus)) + rng.normal(0, 0.3, n)
    pregnant = (rng.random(n) < _sigmoid(logit)).astype(int)

    transfer_id = np.array([f"T{i:06d}" for i in range(1, n + 1)])
    patient_id = np.array([f"P{i:05d}" for i in rng.integers(1, int(n * 0.8) + 1, n)])
    for i in range(n):
        image = render_blastocyst(rng, int(expansion[i]), int(icm[i]), int(te[i]), render_size).image
        cv2.imwrite(str(root / "images" / f"{transfer_id[i]}.png"), cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA))

    clinical = pd.DataFrame({
        "transfer_id": transfer_id, "patient_id": patient_id, "female_age": age, "bmi": bmi, "amh_ng_ml": amh,
        "endometrial_thickness_mm": endometrium, "previous_failed_transfers": failed, "infertility_duration_years": duration,
        "infertility_cause": cause, "frozen_transfer": frozen, "clinical_pregnancy": pregnant,
    })
    manifest = pd.DataFrame({"transfer_id": transfer_id, "image_path": [f"images/{t}.png" for t in transfer_id]})
    reference = pd.DataFrame({"transfer_id": transfer_id, "expansion": expansion + 1, "icm": np.array(list("ABC"))[icm],
                              "te": np.array(list("ABC"))[te], "true_probability": _sigmoid(logit).round(4)})

    # Raw data problems for the processing layer to handle.
    clinical.loc[rng.random(n) < 0.15, "amh_ng_ml"] = np.nan
    clinical.loc[rng.random(n) < 0.08, "endometrial_thickness_mm"] = np.nan
    clinical.loc[rng.random(n) < 0.05, "bmi"] = np.nan
    clinical.loc[rng.random(n) < 0.003, "bmi"] = 999.0
    clinical["infertility_cause"] = np.where(rng.random(n) < 0.04, clinical["infertility_cause"].str.upper(), clinical["infertility_cause"])
    clinical = pd.concat([clinical, clinical.sample(frac=0.004, random_state=seed)], ignore_index=True)       # duplicate submissions
    manifest = manifest.drop(manifest.sample(frac=0.015, random_state=seed).index)                           # images never exported
    clinical = clinical.drop(clinical.sample(frac=0.01, random_state=seed + 1).index).reset_index(drop=True)  # records never linked

    clinical.to_csv(root / "clinical.csv", index=False)
    manifest.to_csv(root / "image_manifest.csv", index=False)
    reference.to_csv(root / "reference_grades.csv", index=False)       # evaluation only: never a model input
    return {"clinical_rows": len(clinical), "images": len(manifest), "pregnancy_rate": float(pregnant.mean())}
