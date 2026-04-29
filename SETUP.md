# Setup Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but inference will be slow)
- Internet connection for first run (HuggingFace model weights + ConvoKit corpus download)

---

## Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

---

## Model Weights

The fine-tuned Longformer weights are **not** committed to the repo (file is too large for git).

Place the weights file at:

```
models/longformer_triple_best.pt
```

If you need to retrain from scratch, run the training notebook (see below).

---

## Running the Web App

```bash
uvicorn src.argument_evaluator:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

The API exposes one endpoint:

| Method | Path | Body |
|---|---|---|
| POST | `/evaluate` | `{"text": "<argument text>"}` |

The response includes `verdict`, `confidence`, `convincing_score`, and `not_convincing_score`.

---

## Running the Notebook

```bash
jupyter notebook "notebooks/cmv_longformer_triple (1).ipynb"
```

The notebook will download the ConvoKit `winning-args-corpus` (~500 MB) on first run and cache it locally. Subsequent runs skip the download.

---

## Project Structure

```
data/
  longformer_triple_best.pt   # fine-tuned weights (not in git)
docs/
  triple_encoding_design.md   # design decision notes
models/
  longform.py                 # LongformerTripleClassifier class + build_model()
notebooks/
  cmv_longformer_triple.ipynb # training + evaluation
src/
  argument_evaluator.py       # FastAPI app
  static/
    index.html                # frontend UI
```

---

## Design Notes

See [docs/triple_encoding_design.md](docs/triple_encoding_design.md) for an explanation of why triple encoding (packing both arguments into one sequence) was chosen over separate encoding, and what evaluation metrics are meaningful for this task.
