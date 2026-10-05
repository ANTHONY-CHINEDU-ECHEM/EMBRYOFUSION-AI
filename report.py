"""Figures for the README, rebuilt from saved predictions and metrics."""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from embryofusion.config import Config, resolve

INK, VIOLET, TEAL, SUN, STONE = "#1D1B2F", "#6C3FC5", "#168A8A", "#E8A317", "#9A9CA8"
LABELS = {"tabular_only": "Clinical record only", "image_only_cnn": "Embryo image only", "late_fusion": "Late fusion (stacked)",
          "fusion_cnn": "Joint fusion, CNN", "fusion_vit": "Joint fusion, ViT"}
COLOURS = {"tabular_only": TEAL, "image_only_cnn": SUN, "late_fusion": STONE, "fusion_cnn": VIOLET, "fusion_vit": INK}


def _style() -> None:
    plt.rcParams.update({"savefig.dpi": 160, "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.grid": True, "grid.alpha": 0.25})


def generate_figures(cfg: Config) -> list[str]:
    _style()
    report_dir, figure_dir, root = resolve(cfg.artifacts.report_dir), resolve(cfg.artifacts.figure_dir), resolve(cfg.data.root)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((report_dir / "metrics.json").read_text())
    preds = pd.read_csv(report_dir / "test_predictions.csv")
    y = preds["clinical_pregnancy"].to_numpy()
    names = [n for n in LABELS if n in report["results"]]
    selected = report["selected_fusion"]
    written = []

    # 1. ROC curves.
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    for name in names:
        fpr, tpr, _ = roc_curve(y, preds[name])
        ax.plot(fpr, tpr, color=COLOURS[name], lw=2.4 if name == selected else 1.5, ls="--" if name == "late_fusion" else "-",
                label=f"{LABELS[name]}  {report['results'][name]['test']['roc_auc']:.3f}")
    fpr, tpr, _ = roc_curve(y, preds["true_probability"])
    ax.plot(fpr, tpr, color=INK, lw=1, ls=":", label=f"Simulator ceiling  {report['simulator_ceiling_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=INK, lw=0.7, alpha=0.4)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Clinical pregnancy prediction on unseen patients")
    ax.legend(title="Test ROC AUC", frameon=False, loc="lower right", fontsize=9.3)
    fig.tight_layout(); fig.savefig(figure_dir / "roc_curves.png"); plt.close(fig); written.append("roc_curves.png")

    # 2. AUC with bootstrap intervals.
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    for i, name in enumerate(names):
        test = report["results"][name]["test"]
        low, high = test["roc_auc_ci"]
        ax.errorbar(test["roc_auc"], i, xerr=[[test["roc_auc"] - low], [high - test["roc_auc"]]], fmt="o", color=COLOURS[name], ms=9, capsize=4, lw=2)
        ax.text(high + 0.004, i, f"{test['roc_auc']:.3f}", va="center", fontsize=9.5)
    ax.axvline(report["simulator_ceiling_auc"], color=INK, ls=":", lw=1)
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names]); ax.invert_yaxis()
    ax.set(xlabel="Test ROC AUC with 95% bootstrap interval", title="Two views beat one")
    fig.tight_layout(); fig.savefig(figure_dir / "auc_intervals.png"); plt.close(fig); written.append("auc_intervals.png")

    # 3. Modality ablation inside the fusion model.
    ablation = report["modality_ablation"]
    keys = [("full", "Both views"), ("image_shuffled", "Image swapped with another patient's"), ("image_hidden", "Image withheld"),
            ("tabular_shuffled", "Record swapped with another patient's"), ("tabular_hidden", "Record withheld")]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    values = [ablation[k] for k, _ in keys]
    ax.barh([label for _, label in keys], values, color=[VIOLET, SUN, SUN, TEAL, TEAL])
    for i, v in enumerate(values):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9.5)
    ax.invert_yaxis(); ax.set(xlim=(0.45, max(values) + 0.05), xlabel="Test ROC AUC of the fusion model", title="How much the fusion model relies on each view")
    fig.tight_layout(); fig.savefig(figure_dir / "modality_ablation.png"); plt.close(fig); written.append("modality_ablation.png")

    # 4. Performance by age band and by embryo quality.
    preds["age_band"] = pd.cut(preds["female_age"], [0, 34.99, 37.99, 50], labels=["Under 35", "35 to 37", "38 and over"])
    preds["embryo_group"] = np.where((preds["icm"] != "C") & (preds["te"] != "C"), "Good embryo (no C grade)", "Poorer embryo (any C grade)")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for ax, column, title in [(axes[0], "age_band", "By maternal age"), (axes[1], "embryo_group", "By embryo quality (reference grades)")]:
        groups = [g for g in preds[column].dropna().unique()]
        groups = sorted(groups, key=lambda g: list(preds[column].cat.categories).index(g)) if column == "age_band" else sorted(groups)
        width = 0.26
        for j, name in enumerate(["tabular_only", "image_only_cnn", selected]):
            aucs = [roc_auc_score(preds.loc[preds[column] == g, "clinical_pregnancy"], preds.loc[preds[column] == g, name]) for g in groups]
            ax.bar(np.arange(len(groups)) + (j - 1) * width, aucs, width, color=COLOURS[name], label=LABELS[name])
        ax.set_xticks(range(len(groups)), groups); ax.set(ylim=(0.5, 0.8), ylabel="ROC AUC", title=title)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figure_dir / "subgroup_auc.png"); plt.close(fig); written.append("subgroup_auc.png")

    # 5. Example cases: what each view says and what the fusion concludes.
    rng = np.random.default_rng(cfg.seed)
    disagreement = (preds["tabular_only"] - preds["image_only_cnn"]).abs()
    chosen = preds.loc[disagreement.sort_values(ascending=False).index[:60]].sample(8, random_state=int(rng.integers(0, 1000)))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 7.2))
    for ax, (_, row) in zip(axes.ravel(), chosen.iterrows()):
        ax.imshow(cv2.imread(str(root / row["image_path"]), cv2.IMREAD_GRAYSCALE), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        endometrium = "not recorded" if pd.isna(row["endometrial_thickness_mm"]) else f"{row['endometrial_thickness_mm']:.1f} mm"
        ax.set_title(f"Age {row['female_age']:.0f}, endometrium {endometrium}\nrecord {100 * row['tabular_only']:.0f}%   image {100 * row['image_only_cnn']:.0f}%   "
                     f"fused {100 * row[selected]:.0f}%\noutcome: {'pregnant' if row['clinical_pregnancy'] else 'not pregnant'}", fontsize=9.3, loc="left")
    fig.suptitle("Cases where the two views disagree most", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(figure_dir / "example_cases.png"); plt.close(fig); written.append("example_cases.png")

    # 6. Training curves from the tracker mirror.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name in [n for n in names if n != "late_fusion"]:
        path = report_dir / "tracking" / f"run_{name}.json"
        if path.exists():
            history = json.loads(path.read_text())["history"]
            ax.plot([h["step"] for h in history], [h["val_auc"] for h in history], "o-", color=COLOURS[name], label=LABELS[name])
    ax.set(xlabel="Epoch", ylabel="Validation ROC AUC", title="Runs logged to Weights and Biases")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(figure_dir / "training_curves.png"); plt.close(fig); written.append("training_curves.png")
    return written
