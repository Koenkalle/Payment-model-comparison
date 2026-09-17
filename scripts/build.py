"""Build registered, self-contained payment tools from shared source files."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from framework.registry import browser_scripts, manifest as plugin_manifest


def read_json(name):
    return json.loads((ROOT / name).read_text())


def source(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing or invalid tool source: {name}")
    return path.read_text()


def validate_registry(registry):
    ids, outputs = set(), set()
    for tool in registry["tools"]:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", tool["id"]) or tool["id"] in ids:
            raise ValueError("Tool IDs must be unique lowercase names.")
        if (
            not re.fullmatch(r"[a-z][a-z0-9-]*\.html", tool["output"])
            or tool["output"] in outputs
        ):
            raise ValueError("Tool outputs must be unique root HTML filenames.")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", tool["model_data_id"]):
            raise ValueError("Invalid model-data element ID.")
        ids.add(tool["id"])
        outputs.add(tool["output"])
        source(tool["template"])
        for script in registry["shared_scripts"] + tool["scripts"]:
            source(script)


def json_script(identifier, payload):
    serialized = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return f'<script type="application/json" id="{identifier}">{serialized}</script>\n'


def navigation(registry, active):
    links = []
    for tool in registry["tools"]:
        current = ' aria-current="page"' if tool["id"] == active else ""
        links.append(
            f'<a href="{html.escape(tool["output"], quote=True)}"{current}>{html.escape(tool["label"])}</a>'
        )
    return source("shared/ui/navigation.html").replace(
        "<!-- TOOL_LINKS -->", "".join(links)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool", default="all", help="Registered tool ID, or all (default)."
    )
    parser.add_argument(
        "--legacy-fragment",
        action="store_true",
        help="Also write the comparison embed fragment beside this repository.",
    )
    args = parser.parse_args()
    registry, catalog = read_json("tools/registry.json"), read_json("designs.json")
    registry["shared_scripts"] = browser_scripts() + registry["shared_scripts"]
    validate_registry(registry)
    selected = [tool for tool in registry["tools"] if args.tool in ("all", tool["id"])]
    if not selected:
        parser.error(f"Unknown tool: {args.tool}")
    subprocess.run(
        ["node", "scripts/tune_policy.js", "--if-stale"], cwd=ROOT, check=True
    )
    if any(tool.get("explanation") for tool in selected):
        subprocess.run(
            [sys.executable, "scripts/export_xgboost_explanations.py"],
            cwd=ROOT,
            check=True,
        )
    validation = read_json("policy-validation.json")
    models = [
        read_json(entry["checkpoint"])
        for entry in plugin_manifest("models")["models"]
        if entry.get("browser") and entry.get("checkpoint")
    ]
    descriptors = {entry["id"]: entry for entry in plugin_manifest("models")["models"]}
    for model in models:
        model["implementation"] = descriptors[model["id"]]
        model["label"] = descriptors[model["id"]]["label"]
        model["policy_validation"] = validation["models"].get(model["id"], {})
    xgb = next(model for model in models if model.get("family") == "xgboost")
    bundle = {
        "version": 5,
        "default": catalog["default"],
        "policy": {
            "training_mode": "unsupervised",
            "decision_policy": "shared",
            "validation": validation["provenance"],
            "warmup": catalog.get("default_warmup", 128),
            "fraud_flags": xgb["training"].get("fraud_flags_requested", 96),
        },
        "models": models,
    }
    (ROOT / "model-bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    ui_scripts = "\n".join(
        "<script>\n" + source(name) + "\n</script>"
        for name in registry.get("ui_scripts", [])
    )
    shell = source("shared/ui/standalone-shell.html").replace(
        "<!-- SHARED_UI_SCRIPTS -->", ui_scripts
    )
    for tool in selected:
        payload = dict(bundle)
        if tool["model_ids"] != "all":
            payload["models"] = [
                model for model in models if model["id"] in tool["model_ids"]
            ]
            if len(payload["models"]) != len(tool["model_ids"]):
                raise ValueError(f'Unknown checkpoint in {tool["id"]}')
            payload["default"] = payload["models"][0]["id"]
        if tool.get("explanation"):
            payload["explanation"] = read_json(tool["explanation"])
        if tool.get("native_models"):
            payload["native_models"] = [
                descriptors[identifier] for identifier in tool["native_models"]
            ]
        template = source(tool["template"])
        # Explicit marker supports any root element; legacy templates use their closing root div.
        data = json_script(tool["model_data_id"], payload)
        if "<!-- TOOL_DATA -->" in template:
            fragment = template.replace("<!-- TOOL_DATA -->", data)
        else:
            at = max(template.rfind("</div>"), template.rfind("</main>"))
            if at < 0:
                raise ValueError(
                    f'Tool template needs a data marker or root: {tool["template"]}'
                )
            fragment = template[:at] + data + template[at:]
        for script in registry["shared_scripts"] + tool["scripts"]:
            fragment += "\n<script>\n" + source(script) + "\n</script>\n"
        page = shell.replace(
            "<!-- FRAUD_DEMO_FRAGMENT -->", navigation(registry, tool["id"]) + fragment
        )
        page = re.sub(
            r"<title>.*?</title>",
            lambda _: "<title>" + html.escape(tool["label"]) + "</title>",
            page,
            count=1,
        )
        (ROOT / tool["output"]).write_text(page)
        if args.legacy_fragment and tool.get("legacy_fragment"):
            (ROOT.parent / tool["legacy_fragment"]).write_text(fragment)
        print(ROOT / tool["output"])
    # Track source and deliverable hashes, excluding local caches and browser artifacts.
    paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path != ROOT / "manifest.json"
        and path.suffix != ".pyc"
        and "node_modules" not in path.parts
        and "artifacts" not in path.parts
        and "test-artifacts" not in path.parts
        and ".vscode" not in path.parts
    ]
    manifest = {
        "version": 12,
        "file_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
