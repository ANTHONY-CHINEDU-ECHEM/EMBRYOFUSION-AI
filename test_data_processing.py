"""Unit tests for the data processing layer. These run on every push in continuous integration."""
import numpy as np
import pandas as pd
import pytest

from embryofusion.data import processing as dp

pytestmark = pytest.mark.data


def clinical_rows(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "transfer_id": [f"T{i}" for i in range(n)], "patient_id": [f"P{i // 2}" for i in range(n)],
        "female_age": [30, 34, 37, 39, 41, 33][:n], "bmi": [22.0, 24.5, 27.0, 31.0, 23.5, 26.0][:n],
        "amh_ng_ml": [3.1, 2.2, np.nan, 0.9, 0.6, 2.8][:n], "endometrial_thickness_mm": [9.5, 10.2, 8.1, np.nan, 7.4, 11.0][:n],
        "previous_failed_transfers": [0, 1, 0, 2, 3, 0][:n], "infertility_duration_years": [1.5, 2.0, 3.5, 4.0, 6.0, 2.5][:n],
        "infertility_cause": ["unexplained", "TUBAL", "male_factor", "ovulatory", "endometriosis", " Unexplained "][:n],
        "frozen_transfer": [1, 0, 1, 1, 0, 1][:n], "clinical_pregnancy": [1, 0, 1, 0, 0, 1][:n],
    })


def test_schema_validation_reports_missing_columns():
    with pytest.raises(dp.SchemaError, match="missing columns"):
        dp.validate_schema(clinical_rows().drop(columns=["bmi", "female_age"]))


def test_schema_validation_rejects_bad_outcomes_and_missing_ids():
    bad_outcome = clinical_rows()
    bad_outcome.loc[0, "clinical_pregnancy"] = 2
    with pytest.raises(dp.SchemaError, match="0 or 1"):
        dp.validate_schema(bad_outcome)
    no_id = clinical_rows()
    no_id.loc[1, "transfer_id"] = None
    with pytest.raises(dp.SchemaError, match="transfer_id"):
        dp.validate_schema(no_id)


def test_cleaning_removes_duplicates_and_normalises_categories():
    raw = pd.concat([clinical_rows(), clinical_rows().iloc[[0, 2]]], ignore_index=True)
    audit = dp.ProcessingAudit()
    clean = dp.clean_clinical(raw, audit)
    assert len(clean) == 6 and clean["transfer_id"].is_unique
    assert set(clean["infertility_cause"]) <= set(dp.CAUSES)
    assert audit.counts["duplicate_transfers_removed"] == 2 and audit.counts["cause_labels_normalised"] == 2


def test_cleaning_blanks_impossible_values_but_keeps_the_row():
    raw = clinical_rows()
    raw.loc[0, "bmi"] = 999.0
    raw.loc[1, "female_age"] = 7
    raw.loc[2, "infertility_cause"] = "martian"
    audit = dp.ProcessingAudit()
    clean = dp.clean_clinical(raw, audit)
    assert len(clean) == 6 and np.isnan(clean.loc[0, "bmi"]) and np.isnan(clean.loc[1, "female_age"])
    assert clean.loc[2, "infertility_cause"] == "unexplained"
    assert audit.counts["out_of_range_bmi"] == 1 and audit.counts["unknown_causes_mapped"] == 1


def test_cleaning_does_not_mutate_its_input():
    raw = clinical_rows()
    before = raw.copy()
    dp.clean_clinical(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_pairing_is_an_inner_join_and_counts_orphans():
    clinical = dp.clean_clinical(clinical_rows())
    manifest = pd.DataFrame({"transfer_id": ["T0", "T1", "T2", "T2", "T99"], "image_path": ["a.png", "b.png", "c.png", "c2.png", "z.png"]})
    audit = dp.ProcessingAudit()
    paired = dp.pair_modalities(clinical, manifest, audit=audit)
    assert paired["transfer_id"].tolist() == ["T0", "T1", "T2"] and paired.loc[2, "image_path"] == "c.png"
    assert audit.counts == {"clinical_records_without_image": 3, "images_without_clinical_record": 1}


def test_pairing_drops_rows_whose_image_file_is_missing(tmp_path):
    clinical = dp.clean_clinical(clinical_rows(2))
    (tmp_path / "present.png").write_bytes(b"x")
    manifest = pd.DataFrame({"transfer_id": ["T0", "T1"], "image_path": ["present.png", "absent.png"]})
    audit = dp.ProcessingAudit()
    paired = dp.pair_modalities(clinical, manifest, tmp_path, audit)
    assert paired["transfer_id"].tolist() == ["T0"] and audit.counts["image_files_missing_on_disk"] == 1


def test_patient_split_has_no_leakage_and_is_deterministic():
    frame = pd.DataFrame({"patient_id": np.repeat(np.arange(200), 2), "x": np.arange(400)})
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    first, second = dp.patient_split(frame, fractions, seed=1), dp.patient_split(frame, fractions, seed=1)
    assert first.equals(second) and not first.isna().any()
    assert (frame.assign(split=first).groupby("patient_id")["split"].nunique() == 1).all()
    shares = first.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.7, abs=0.02) and shares["test"] == pytest.approx(0.15, abs=0.02)
    assert not first.equals(dp.patient_split(frame, fractions, seed=2))


def test_patient_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        dp.patient_split(pd.DataFrame({"patient_id": [1, 2]}), {"train": 0.5, "val": 0.1, "test": 0.1}, seed=0)


def test_tabular_preprocessor_imputes_flags_and_encodes():
    clean = dp.clean_clinical(clinical_rows())
    features = dp.NUMERIC + dp.BINARY + dp.CATEGORICAL
    preprocessor = dp.build_tabular_preprocessor().fit(clean[features])
    matrix = preprocessor.transform(clean[features])
    assert not np.isnan(matrix).any()
    assert matrix.shape == (6, len(dp.NUMERIC) + 2 + 1 + len(dp.CAUSES))      # two columns had gaps, so two indicator flags
    unseen = clean[features].head(1).assign(infertility_cause="brand_new")
    assert preprocessor.transform(unseen)[0, -len(dp.CAUSES):].sum() == 0


def test_image_loading_and_normalisation(tmp_path):
    import cv2

    image = np.random.default_rng(0).integers(0, 255, (50, 50), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "e.png"), image)
    loaded = dp.load_image(tmp_path / "e.png", 32)
    assert loaded.shape == (32, 32) and loaded.dtype == np.float32 and 0 <= loaded.min() and loaded.max() <= 1
    normalised = dp.normalise_image(loaded)
    assert abs(normalised.mean()) < 1e-4 and normalised.std() == pytest.approx(1.0, abs=1e-3)
    assert np.isfinite(dp.normalise_image(np.zeros((8, 8), np.float32))).all()      # blank frame must not divide by zero
    with pytest.raises(FileNotFoundError):
        dp.load_image(tmp_path / "missing.png", 32)


def test_full_processing_chain_on_generated_data(tiny_root):
    paired, images, audit = dp.prepare_dataset(tiny_root, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0, image_size=32)
    assert len(paired) == len(images) == audit["paired_transfers"] and images.shape[1:] == (32, 32)
    assert paired["transfer_id"].is_unique and set(paired["split"]) == {"train", "val", "test"}
    assert audit["clinical_records_without_image"] > 0 and audit["images_without_clinical_record"] > 0
    assert "true_probability" not in paired.columns and "icm" not in paired.columns     # reference grades never leak into inputs
