# TabFM in the model trainer

`tabfm` uses Google's pretrained TabFM 1.0 classification model through its
official Python package and PyTorch backend. It accepts the existing
`numeric-table/v1` dataset view and returns the probability of fraud (`label=1`).
The integration supports binary fraud classification and saved-run comparison.

TabFM's weights stay frozen. **Prepare and save model** selects labelled examples
from the chronological training partition and supplies them as context. The
normal validation partition selects the probability cutoff, and the test
partition supplies the held-out report. Unknown training outcomes and outcomes
reported after the training boundary are excluded by the existing pipeline.

## Install and run

Complete the [README setup](../README.md#setup), then install TabFM in the same
environment and start (or restart) the server:

```sh
python -m pip install -r requirements-tabfm.txt
python serve.py
```

Open <http://127.0.0.1:8000/trainer.html>, select a dataset with both known
classes, choose **TabFM (Google foundation model)**, then **Prepare and save model**.

To use `serve-tabfm.sh` on Linux, macOS, or WSL, create its environment once
from the project folder (Python 3.11+ and Git required):

```sh
tabfm_env="$HOME/.local/share/payment-model-comparison/tabfm-env"
python3 -m venv "$tabfm_env"
"$tabfm_env/bin/python" -m pip install -r requirements-models.txt -r requirements-temporal.txt -r requirements-tabfm.txt
bash serve-tabfm.sh
```

Next time, just run `bash serve-tabfm.sh`. It caches weights in
`artifacts/tabfm/hf-cache`; downloads and environments are not included in Git.
Pass server options as usual, for example `bash serve-tabfm.sh --port 8001`.
To reuse another environment, set `PAYMENT_TABFM_PYTHON` to its Python executable.

The optional requirements pin the Google source revision
`fbb665569425fd2f490c6576b3af967876fe11ff` (package version 1.0.1). The adapter pins
the classification checkpoint in `google/tabfm-1.0.0-pytorch` to revision
`77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83`.

The first preparation downloads the classification configuration and weights
from Hugging Face into its normal local cache. The weights alone are about
6.1 GiB, and CPU inference also requires working memory beyond the checkpoint.
Context and prediction batch limits control additional memory, but cannot reduce
the size of the model itself. Allow time for the initial download and inference.
Subsequent preparations and comparisons reuse cached weights. `HF_HOME` can
point to a cache directory with enough space; a prepopulated cache also supports
Hugging Face's `HF_HUB_OFFLINE=1` mode. Dataset rows are processed locally.

Allow more RAM than the weight size: a small local CPU check used about
10.5 GiB. The prepared run appears alongside the other saved models.

The CLI supports the same adapter:

```sh
python experiment.py train --config examples/tabfm-experiment.json --output artifacts/my-tabfm
python experiment.py evaluate --config examples/tabfm-dataset.config.json --artifact artifacts/my-tabfm --partition test --output artifacts/tabfm-test.json
```

The example uses the bundled invented Handbook fixture, which is suitable for
checking integration rather than measuring benchmark quality.

## Controls and saved state

| Control | Default | Meaning |
| --- | --- | --- |
| Maximum context rows | 100 | Deterministic sample of known training rows, retaining both classes; range 2–2048 |
| Ensemble members | 1 | Number of upstream inference ensemble members; range 1–8 |
| Prediction batch size | 128 | Maximum query rows per inference call; range 1–256 |
| Random seed | 42 | Reproducible context selection and upstream preprocessing |
| Fixed fraud probability cutoff | Blank | Validation-selected cutoff, or an explicitly supplied probability |

Inference runs on CPU in float32 with two PyTorch threads, matching the web
pipeline worker. At most 500 numeric features are accepted. Feature order must
match the saved model. Increasing
context size and ensemble members increases computation. Upstream pads small
query batches to 128 rows, so reducing that setting can increase repeated work.
Numeric preprocessing is fitted by TabFM using the selected training context.

The model artifact stores the selected labelled context, feature order, inference
parameters, library versions, and pretrained model identity using JSON, with the
existing artifact checksum. It reconstructs the upstream classifier on reload using the pinned
weights. Reloading requires the recorded library versions to keep preprocessing
and inference reproducible. These artifacts contain training data; retain them
under the same access controls as the source dataset. They require the optional TabFM dependencies and
cached or downloadable weights when evaluated. Pretrained weights are not copied
into each run.

Comparisons use the existing common held-out population and saved cutoff.
TabFM does not provide the additive feature explanations used by XGBoost, and
it is available through Python saved-run evaluation rather than browser inference.

## Upstream sources and licensing

Google's source code is Apache-2.0 licensed. Its released pretrained weights use
the separate **TabFM Non-Commercial License v1.0**, which restricts them to
non-commercial, non-production use. This restriction is also shown beside the
model controls.

- [Introduction and architecture](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)
- [Official source and installation](https://github.com/google-research/tabfm)
- [Model card and checkpoint files](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- [Pretrained weight license](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)
