"""CLI for independently registered datasets and real model implementations."""
import argparse, json, logging
from pathlib import Path
from framework.registry import manifest, load_dataset
from framework.experiments import train_experiment, evaluate_artifact
from framework.contracts import EventDataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "list", help="List model and dataset implementations and capabilities."
    )
    inspect = sub.add_parser(
        "inspect-data",
        help="Inspect source capabilities, ordering, provenance and label coverage.",
    )
    inspect.add_argument("--config", required=True, type=Path)
    for command in ("prepare", "train", "evaluate", "export-fraud"):
        child = sub.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
        if command == "evaluate":
            child.add_argument("--artifact", required=True, type=Path)
            child.add_argument(
                "--partition",
                choices=[
                    "all",
                    "train",
                    "validation",
                    "model_validation",
                    "policy_validation",
                    "test",
                ],
                default="all",
            )
    args = parser.parse_args()
    try:
        if args.command == "list":
            for kind in ("models", "datasets"):
                for entry in manifest(kind)[kind]:
                    print(
                        kind,
                        entry["id"],
                        entry.get("execution", entry.get("schema")),
                        entry.get("status", entry.get("origin")),
                    )
            from datasets.adapters import source_catalog

            for source in source_catalog():
                print(
                    "sources",
                    source["dataset_id"],
                    "entities=" + (",".join(source["entity_types"]) or "none"),
                    "features=" + ",".join(source["feature_names"]),
                )
            return
        config = json.loads(args.config.read_text())
        base = args.config.resolve().parent
        if args.command == "inspect-data":
            from datasets.inspection import inspect_dataset

            print(json.dumps(inspect_dataset(config, base), indent=2, allow_nan=False))
            return
        if args.command == "train":
            _, result = train_experiment(config, args.output, base)
            print(json.dumps(result["metrics"], indent=2))
            return
        if args.output.exists():
            raise ValueError("Output already exists; choose a new file.")
        if args.command == "export-fraud":
            from framework.fraud_export import export_fraud

            output = export_fraud(config, base)
        elif args.command == "prepare":
            if (
                config.get("loader") == "fraud_dataset"
                and config.get("view", "stream") == "stream"
            ):
                if config.get("selection"):
                    raise ValueError(
                        "Prepare the complete stream, then select an interval in a numeric, graph or payments view."
                    )
                from datasets.stream import prepare_source

                with prepare_source(config, base, args.output):
                    pass
                print(args.output)
                return
            data = load_dataset(config, base)
            if not isinstance(data, EventDataset):
                if hasattr(data, "close"):
                    data.close()
                raise ValueError(
                    'prepare exports view="payments" to JSON or fraud_dataset view="stream" to SQLite; numeric and graph views feed the experiment runner directly.'
                )
            output = data.document
        else:
            output = evaluate_artifact(args.artifact, config, base, args.partition)
        serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(serialized)
        print(args.output)
    except ImportError as error:
        parser.exit(
            2,
            str(error)
            + "; install requirements-temporal.txt for graph models or requirements-models.txt for tabular models.\n",
        )
    except (
        ValueError,
        KeyError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
    ) as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
