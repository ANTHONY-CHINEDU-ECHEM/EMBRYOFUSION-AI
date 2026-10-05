"""Experiment tracking with Weights and Biases, with a local fallback.

The default mode is ``offline``: runs are written to the ``wandb`` folder and
can be uploaded later with ``wandb sync``. Set ``tracking.mode`` to ``online``
after ``wandb login`` to stream runs, or to ``disabled`` to skip W&B entirely.
Whatever the mode, every run is also mirrored to a JSON file so the report
step never depends on an external service.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, project: str, name: str, config: dict, mode: str, log_dir: Path):
        self.name, self.history, self.summary = name, [], {}
        self.path = Path(log_dir) / f"run_{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.run = None
        if mode != "disabled":
            try:
                import wandb

                os.environ.setdefault("WANDB_SILENT", "true")
                self.run = wandb.init(project=project, name=name, config=config, mode=mode, reinit=True, dir=str(log_dir))
            except Exception as exc:  # pragma: no cover - depends on the environment
                logger.warning("Weights and Biases unavailable (%s); logging locally only.", exc)

    def log(self, metrics: dict, step: int) -> None:
        self.history.append({"step": step, **metrics})
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self, summary: dict) -> None:
        self.summary = summary
        if self.run is not None:
            for key, value in summary.items():
                self.run.summary[key] = value
            self.run.finish()
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"name": self.name, "config": self.config, "history": self.history, "summary": summary}, handle, indent=2, default=float)
