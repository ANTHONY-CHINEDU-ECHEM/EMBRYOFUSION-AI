"""Command line entry point: ``python -m embryofusion.cli <command>``."""
from __future__ import annotations

import argparse
import json
import logging

from embryofusion.config import load_config, resolve


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="embryofusion", description="EmbryoFusion AI pipeline")
    parser.add_argument("command", choices=["generate", "train", "report", "all", "predict"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--record", default=None, help="JSON clinical record for the predict command")
    parser.add_argument("--image", default=None, help="Embryo image path for the predict command")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config)

    if args.command in {"generate", "all"}:
        from embryofusion.data.simulate import generate_paired_dataset

        print(json.dumps(generate_paired_dataset(resolve(cfg.data.root), cfg.data.n_transfers, cfg.data.render_size, cfg.data.image_size, cfg.seed)))
    if args.command in {"train", "all"}:
        from embryofusion.training.pipeline import run_pipeline

        report = run_pipeline(cfg)
        print(json.dumps({name: round(r["test"]["roc_auc"], 4) for name, r in report["results"].items()}, indent=2))
    if args.command in {"report", "all"}:
        from embryofusion.evaluation.report import generate_figures

        print("Figures:", ", ".join(generate_figures(cfg)))
    if args.command == "predict":
        from embryofusion.inference.predict import FusionPredictor

        print(json.dumps(FusionPredictor().predict(json.loads(args.record), args.image), indent=2))


if __name__ == "__main__":
    main()
