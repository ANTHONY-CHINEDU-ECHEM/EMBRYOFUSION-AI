"""Train the single modality baselines and the fusion models, then compare them."""
from __future__ import annotations

import copy
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn

from embryofusion import __version__
from embryofusion.config import Config, resolve
from embryofusion.data.processing import BINARY, CATEGORICAL, NUMERIC, TARGET, build_tabular_preprocessor, prepare_dataset
from embryofusion.evaluation.metrics import paired_auc_difference, summary_metrics
from embryofusion.models.fusion import FusionNet
from embryofusion.training.tracker import Tracker

logger = logging.getLogger(__name__)


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Flips and quarter turns: an embryo has no canonical orientation."""
    if torch.rand(1, generator=generator) < 0.5:
        images = images.flip(-1)
    if torch.rand(1, generator=generator) < 0.5:
        images = images.flip(-2)
    return torch.rot90(images, int(torch.randint(0, 4, (1,), generator=generator)), dims=(-2, -1))


@torch.no_grad()
def predict(model: FusionNet, tabular: torch.Tensor, images: torch.Tensor, hide: str | None = None, batch: int = 256) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(tabular[i:i + batch], images[i:i + batch], hide=hide)) for i in range(0, len(tabular), batch)]
    return torch.cat(out).numpy()


def train_model(model: FusionNet, data: dict, cfg: Config, tracker: Tracker) -> tuple[FusionNet, int]:
    """Fit one model with early stopping on validation AUC."""
    generator = torch.Generator().manual_seed(cfg.seed)
    tr = cfg.training
    tab, img, y = data["train"]
    optimiser = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    steps = tr.epochs * int(np.ceil(len(y) / tr.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=tr.learning_rate, total_steps=steps, pct_start=0.2)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, tr.epochs + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        total = 0.0
        for start in range(0, len(order), tr.batch_size):
            index = order[start:start + tr.batch_size]
            if len(index) < 2:
                continue
            images = augment(img[index], generator) if model.image is not None else img[index]
            loss = loss_fn(model(tab[index], images), y[index])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimiser.step()
            scheduler.step()
            total += float(loss.detach()) * len(index)
        val_auc = float(roc_auc_score(data["val"][2].numpy(), predict(model, data["val"][0], data["val"][1])))
        tracker.log({"train_loss": total / len(y), "val_auc": val_auc, "learning_rate": scheduler.get_last_lr()[0]}, step=epoch)
        logger.info("%s epoch %d | loss %.4f | val AUC %.4f", tracker.name, epoch, total / len(y), val_auc)
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, best_epoch, stale = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= tr.patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def run_pipeline(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    root = resolve(cfg.data.root)
    paired, images, audit = prepare_dataset(root, dict(cfg.data.split), cfg.seed, cfg.data.image_size)
    preprocessor = build_tabular_preprocessor().fit(paired.loc[paired["split"] == "train", NUMERIC + BINARY + CATEGORICAL])
    tabular = preprocessor.transform(paired[NUMERIC + BINARY + CATEGORICAL]).astype(np.float32)
    data, frames = {}, {}
    for split in ["train", "val", "test"]:
        mask = (paired["split"] == split).to_numpy()
        frames[split] = paired[mask].reset_index(drop=True)
        data[split] = (torch.from_numpy(tabular[mask]), torch.from_numpy(images[mask]).unsqueeze(1),
                       torch.from_numpy(paired.loc[mask, TARGET].to_numpy(dtype=np.float32)))
    logger.info("Paired transfers: %s", {k: len(v) for k, v in frames.items()})

    model_dir, report_dir = resolve(cfg.artifacts.model_dir), resolve(cfg.artifacts.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    y_val, y_test = data["val"][2].numpy(), data["test"][2].numpy()
    rounds = cfg.evaluation.bootstrap_rounds
    results, val_pred, test_pred, models = {}, {}, {}, {}

    for experiment in cfg.training.experiments:
        experiment = dict(experiment)
        name, started = experiment["name"], time.time()
        torch.manual_seed(cfg.seed)
        model = FusionNet(experiment["mode"], tabular.shape[1], cfg.data.image_size, dict(cfg.model), experiment.get("image_encoder", "cnn"))
        parameters = sum(p.numel() for p in model.parameters())
        tracker = Tracker(cfg.tracking.project, name, {**experiment, **dict(cfg.training), "parameters": parameters, "seed": cfg.seed},
                          cfg.tracking.mode, report_dir / "tracking")
        model, best_epoch = train_model(model, data, cfg, tracker)
        val_pred[name], test_pred[name] = predict(model, *data["val"][:2]), predict(model, *data["test"][:2])
        results[name] = {**experiment, "parameters": parameters, "best_epoch": best_epoch, "train_seconds": round(time.time() - started, 1),
                         "val_roc_auc": float(roc_auc_score(y_val, val_pred[name])), "test": summary_metrics(y_test, test_pred[name], rounds, cfg.seed)}
        tracker.finish({"val_roc_auc": results[name]["val_roc_auc"], **{f"test_{k}": v for k, v in results[name]["test"].items() if k != "roc_auc_ci"}})
        models[name] = model
        logger.info("%s | val AUC %.4f | test AUC %.4f", name, results[name]["val_roc_auc"], results[name]["test"]["roc_auc"])

    # Late fusion comparator: stack the two single modality models with a logistic regression fitted on validation data.
    unimodal = [n for n, r in results.items() if r["mode"] != "fusion"]
    if len(unimodal) >= 2:
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        stacker = LogisticRegression().fit(np.column_stack([logit(val_pred[n]) for n in unimodal]), y_val)
        test_pred["late_fusion"] = stacker.predict_proba(np.column_stack([logit(test_pred[n]) for n in unimodal]))[:, 1]
        results["late_fusion"] = {"name": "late_fusion", "mode": "late_fusion", "test": summary_metrics(y_test, test_pred["late_fusion"], rounds, cfg.seed)}

    fusion_names = [n for n, r in results.items() if r["mode"] == "fusion"]
    selected = max(fusion_names, key=lambda n: results[n]["val_roc_auc"])
    best_single = max(unimodal, key=lambda n: results[n]["val_roc_auc"])
    model = models[selected]
    rng = np.random.default_rng(cfg.seed)
    permutation = torch.from_numpy(rng.permutation(len(y_test)))
    ablation = {
        "full": results[selected]["test"]["roc_auc"],
        "image_hidden": float(roc_auc_score(y_test, predict(model, *data["test"][:2], hide="image"))),
        "tabular_hidden": float(roc_auc_score(y_test, predict(model, *data["test"][:2], hide="tabular"))),
        "image_shuffled": float(roc_auc_score(y_test, predict(model, data["test"][0], data["test"][1][permutation]))),
        "tabular_shuffled": float(roc_auc_score(y_test, predict(model, data["test"][0][permutation], data["test"][1]))),
    }
    comparisons = {f"{selected}_minus_{other}": paired_auc_difference(y_test, test_pred[selected], test_pred[other], rounds, cfg.seed)
                   for other in results if other != selected}

    reference = pd.read_csv(root / "reference_grades.csv")
    test_frame = frames["test"].merge(reference, on="transfer_id", how="left")
    for name, values in test_pred.items():
        test_frame[name] = values
    test_frame.to_csv(report_dir / "test_predictions.csv", index=False)
    ceiling = float(roc_auc_score(y_test, test_frame["true_probability"]))

    report = {"rows": {k: len(v) for k, v in frames.items()}, "pregnancy_rate": float(paired[TARGET].mean()), "processing_audit": audit,
              "results": results, "selected_fusion": selected, "best_single_modality": best_single, "modality_ablation": ablation,
              "paired_comparisons": comparisons, "simulator_ceiling_auc": ceiling}
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)
    torch.save({"state_dict": model.state_dict(), "experiment": {k: results[selected][k] for k in ["name", "mode", "image_encoder"]},
                "n_tabular": int(tabular.shape[1]), "image_size": cfg.data.image_size, "model_cfg": json.loads(json.dumps(dict(cfg.model))),
                "version": __version__}, model_dir / "fusion_model.pt")
    joblib.dump(preprocessor, model_dir / "tabular_preprocessor.joblib")
    return report
