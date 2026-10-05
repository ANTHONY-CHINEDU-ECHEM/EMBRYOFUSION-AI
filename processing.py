"""Data processing: validation, cleaning, modality pairing, patient level splits and tabular preprocessing.

These functions are deliberately small and pure so they can be unit tested in
continuous integration without any model or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID, GROUP, TARGET = "transfer_id", "patient_id", "clinical_pregnancy"
NUMERIC = ["female_age", "bmi", "amh_ng_ml", "endometrial_thickness_mm", "previous_failed_transfers", "infertility_duration_years"]
BINARY = ["frozen_transfer"]
CATEGORICAL = ["infertility_cause"]
CAUSES = ["unexplained", "male_factor", "tubal", "ovulatory", "endometriosis"]
RANGES = {"female_age": (18, 50), "bmi": (15, 55), "amh_ng_ml": (0.01, 25), "endometrial_thickness_mm": (3, 20),
          "previous_failed_transfers": (0, 15), "infertility_duration_years": (0, 25)}
REQUIRED = [ID, GROUP, TARGET, *NUMERIC, *BINARY, *CATEGORICAL]


class SchemaError(ValueError):
    """Raised when a clinical table cannot be used at all."""


@dataclass
class ProcessingAudit:
    """Counts of every correction made, written to the report for traceability."""

    counts: dict = field(default_factory=dict)

    def add(self, key: str, value: int) -> None:
        if value:
            self.counts[key] = self.counts.get(key, 0) + int(value)


def validate_schema(frame: pd.DataFrame) -> None:
    """Fail fast on structural problems that cleaning cannot repair."""
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaError(f"Clinical table is missing columns: {missing}")
    if frame[ID].isna().any():
        raise SchemaError("Clinical table has rows without a transfer_id.")
    outcome = pd.to_numeric(frame[TARGET], errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise SchemaError("clinical_pregnancy must be 0 or 1 for every row.")


def clean_clinical(frame: pd.DataFrame, audit: ProcessingAudit | None = None) -> pd.DataFrame:
    """Remove duplicates, normalise categories and blank values outside plausible ranges."""
    audit = audit or ProcessingAudit()
    validate_schema(frame)
    out = frame.drop_duplicates(subset=ID, keep="first").copy()
    audit.add("duplicate_transfers_removed", len(frame) - len(out))
    cause = out["infertility_cause"].astype(str).str.strip().str.lower()
    audit.add("cause_labels_normalised", int((cause != out["infertility_cause"]).sum()))
    audit.add("unknown_causes_mapped", int((~cause.isin(CAUSES)).sum()))
    out["infertility_cause"] = cause.where(cause.isin(CAUSES), "unexplained")
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(out[column], errors="coerce")
        invalid = values.notna() & ((values < low) | (values > high))
        audit.add(f"out_of_range_{column}", int(invalid.sum()))
        out[column] = values.mask(invalid)
    out["frozen_transfer"] = pd.to_numeric(out["frozen_transfer"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    out[TARGET] = out[TARGET].astype(int)
    return out.reset_index(drop=True)


def pair_modalities(clinical: pd.DataFrame, manifest: pd.DataFrame, image_root: str | Path | None = None,
                    audit: ProcessingAudit | None = None) -> pd.DataFrame:
    """Inner join clinical records with embryo images and report everything that fails to pair."""
    audit = audit or ProcessingAudit()
    manifest = manifest.drop_duplicates(subset=ID, keep="first")
    audit.add("clinical_records_without_image", int((~clinical[ID].isin(manifest[ID])).sum()))
    audit.add("images_without_clinical_record", int((~manifest[ID].isin(clinical[ID])).sum()))
    paired = clinical.merge(manifest[[ID, "image_path"]], on=ID, how="inner", validate="one_to_one")
    if image_root is not None:
        exists = paired["image_path"].map(lambda p: (Path(image_root) / p).is_file())
        audit.add("image_files_missing_on_disk", int((~exists).sum()))
        paired = paired[exists]
    return paired.reset_index(drop=True)


def patient_split(frame: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Assign train, val or test by patient so one woman's transfers never straddle a boundary."""
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.")
    patients = np.sort(frame[GROUP].unique())
    np.random.default_rng(seed).shuffle(patients)
    cut_train = int(len(patients) * fractions["train"])
    cut_val = cut_train + int(len(patients) * fractions["val"])
    assignment = {p: "train" for p in patients[:cut_train]}
    assignment.update({p: "val" for p in patients[cut_train:cut_val]})
    assignment.update({p: "test" for p in patients[cut_val:]})
    return frame[GROUP].map(assignment).rename("split")


def build_tabular_preprocessor() -> ColumnTransformer:
    """Median imputation with missingness flags and scaling for numbers, one hot for categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    categorical = OneHotEncoder(categories=[CAUSES], handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("binary", "passthrough", BINARY), ("categorical", categorical, CATEGORICAL)])


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Read an embryo image as float32 grayscale in [0, 1] at the requested size."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.shape != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def normalise_image(image: np.ndarray) -> np.ndarray:
    """Per image standardisation, which removes exposure differences between microscopes."""
    return (image - image.mean()) / (image.std() + 1e-6)


def prepare_dataset(root: str | Path, fractions: dict, seed: int, image_size: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Run the whole processing chain and return the paired table, the image array and the audit."""
    root = Path(root)
    audit = ProcessingAudit()
    clinical = clean_clinical(pd.read_csv(root / "clinical.csv"), audit)
    paired = pair_modalities(clinical, pd.read_csv(root / "image_manifest.csv"), root, audit)
    paired["split"] = patient_split(paired, fractions, seed)
    images = np.stack([normalise_image(load_image(root / p, image_size)) for p in paired["image_path"]])
    audit.counts["paired_transfers"] = len(paired)
    return paired, images, audit.counts
