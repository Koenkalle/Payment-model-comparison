"""Compatibility CLI and import facade. Implementations live in models/implementations/."""
import argparse, sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.implementations._common import *
from models.implementations._neural import *
from models.implementations.xgboost_numpy import *
from models.implementations._temporal import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["all"] + list(DESIGNS), default="all")
    ap.add_argument("--steps", "--training-steps", dest="steps", type=int, default=140)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--model-seed", type=int, default=512)
    ap.add_argument(
        "--fraud-flags",
        type=int,
        default=int(CATALOG.get("default_fraud_flags", 96)),
        help="Flagged historical fraud outcomes used by the offline XGBoost trainer.",
    )
    ap.add_argument("--fraud-seed", type=int, default=811)
    ap.add_argument("--xgb-trees", type=int, default=32)
    ap.add_argument(
        "--warmup",
        "--calibration-requests",
        "--training-time",
        dest="warmup",
        type=int,
        default=int(CATALOG.get("default_warmup", 128)),
        help="Default number of unlabeled request scores used to initialize the live cutoff.",
    )
    a = ap.parse_args()
    if (
        a.steps < 1
        or a.batch < 1
        or a.fraud_flags < 0
        or a.xgb_trees < 1
        or a.warmup < 1
    ):
        raise SystemExit(
            "steps, batch, fraud-flags, xgb-trees, and warmup must be positive (fraud-flags may be zero only when not training XGBoost)."
        )
    if not (ROOT / "training-data.json").exists():
        subprocess.run(
            [
                "node",
                "-e",
                "require('fs').writeFileSync('training-data.json',JSON.stringify(require('./shared/runtime/scenarios.js').trainingEpisodes()))",
            ],
            cwd=ROOT,
            check=True,
        )
    for folder in ["models", "parity", "parity-supervised"]:
        (ROOT / folder).mkdir(exist_ok=True)
    data = json.loads((ROOT / "training-data.json").read_text())
    designs = list(DESIGNS) if a.design == "all" else [a.design]
    labeled = (
        ensure_labeled_data(a.fraud_flags, a.fraud_seed)
        if a.fraud_flags > 0
        and any(DESIGNS[d].get("family") != "xgboost" for d in designs)
        else None
    )
    from framework.registry import model_entry, load_module

    for design in designs:
        entry = model_entry(design)
        load_module(entry["python_module"]).train(a, data, labeled)


if __name__ == "__main__":
    main()
