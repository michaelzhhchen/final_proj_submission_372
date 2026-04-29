# ATTRIBUTION.md

This file documents the attribution of AI-generated code, external libraries,
datasets, and other resources used in this project, in accordance with course
submission requirements.

---
## 1. AI-Generated Code

This project was developed with the assistance of Claude. Pseudo-code and outlines were coded by hand by the project author, and Claude was used for complete syntax, as well as extensive debugging. Claude was also used towards the end of the project for substantial clean-up of code, removing unnecessary functions from failed trials of other ideas. Consider the following an exhaustive list of components that were wholly or substantially produced or revised with AI assistance (attribution generated using Claude based on previous chatlogs of requests).

**Model architecture (`LongformerTripleClassifier`)**
- Project author originally implemented a DeBERTa-v3-base model. Claude contributed a replacement of DeBERTa-v3-base with `allenai/longformer-base-4096` to encode OP + both arguments in a single forward pass without truncation and to reduce the need for aggressive token removal.
- Implementation of `global_attention_mask` with position 0 set to 1 (CLS token global attention), required for Longformer classification tasks.
- Mean pooling over non-padding tokens, replacing CLS token extraction.
- Switch from separate per-argument encoding to triple encoding `[OP | ARG_A | ARG_B]` in a single sequence to eliminate length-as-feature leakage, where the model was learning to use padding length as a proxy for argument length rather than reading content.

**Dataset (`CMVTripleDataset`)**
- Pre-generated swap decisions at init time using `np.random.default_rng(seed)` to randomise winner/loser position during training, preventing positional bias while keeping val/test reproducible.
- Fixed multiprocessing RNG issue where `self.rng` was being re-seeded identically across DataLoader workers, producing a skewed label distribution. Resolved by moving swap decisions to init time in the main process.
- Fixed `ValueError` caused by passing `None` as first tokenizer argument when `op_text` was empty; replaced with empty string.

**Training infrastructure (`train_model` function)**
- Cosine warmup schedule (`get_cosine_schedule_with_warmup`) with one full epoch of warmup.
- Mixed precision training (fp16) via `torch.cuda.amp.GradScaler` for speed.
- Switch from `MarginRankingLoss` to `CrossEntropyLoss` corresponding to the reformulation from pairwise ranking to binary classification.
- Early stopping fix: warmup epoch (epoch 1) excluded from checkpoint contention after degenerate val accuracy of 1.0 was saved as best checkpoint due to the model predicting all-zeros before real learning began.

**Data loading and splitting**
- Pairwise extraction using `pair_ids` metadata from the ConvoKit winning-args corpus, replacing per-utterance binary classification using the `success` field, which was found to be effectively per-conversation rather than per-utterance (3,047 of 3,051 conversations had identical labels across all replies).
- Group-aware train/val/test split on `conv_id` to prevent conversation-level leakage.
- NaN fill for `op_text` after CSV cache loading, fixing `ValueError` from pandas reading empty strings as NaN.

**Evaluation and baselines**
- Length heuristic baselines (shorter-wins, longer-wins) added alongside TF-IDF + LR to contextualise model performance.

**Hyperparameter search (Section 5)**
- Fresh model instantiated per LR candidate to prevent weight state bleeding across runs.

**Final Things**

The web-app framework was generated using Claude, and the project author. The project author exported the model to longform.py and then imported it into the web-app. 

All AI-assisted code was reviewed, tested against training logs, and accepted or modified by the project author before inclusion.

Preliminary versions of ATTRIBUTION.md and README.md were generated using Claude based on a review of chat history, and then heavily edited by the project author. 

---

## 2. External Libraries

| Library | Version | Purpose | License |
|---|---|---|---|
| PyTorch (`torch`) | ≥2.0 | Model training, tensor operations, DataLoader | BSD-3-Clause |
| Hugging Face Transformers (`transformers`) | ≥4.35 | Longformer model, tokenizer, schedules | Apache 2.0 |
| Hugging Face Accelerate (`accelerate`) | ≥0.24 | Mixed-precision and device management utilities | Apache 2.0 |
| scikit-learn (`sklearn`) | ≥1.3 | `GroupShuffleSplit`, `TfidfVectorizer`, `LogisticRegression`, metrics | BSD-3-Clause |
| ConvoKit (`convokit`) | ≥2.5 | Corpus loading and utterance iteration | MIT |
| NumPy (`numpy`) | ≥1.24 | Numerical operations, seed setting | BSD-3-Clause |
| pandas | ≥2.0 | DataFrame manipulation, CSV caching of splits | BSD-3-Clause |
| Matplotlib (`matplotlib`) | ≥3.7 | Training curve and error analysis plots | PSF/BSD |
| Seaborn (`seaborn`) | ≥0.12 | Confusion matrix heatmap, plot theming | BSD-3-Clause |

All libraries were installed via `pip` and are publicly available on PyPI.

---

## 3. Dataset

**Winning Arguments Corpus (CMV)**

- **Source:** ConvoKit, distributed by the Cornell Computational Linguistics Lab
- **URL:** https://convokit.cornell.edu/documentation/winning_args_corpus.html
- **Underlying data:** Reddit community r/ChangeMyView (r/CMV), sourced via
  the Pushshift Reddit API
- **Original paper:**
  Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016).
  Winning Arguments: Interaction Dynamics and Persuasion Strategies in
  Good-Faith Online Discussions. *Proceedings of WWW 2016*.
  https://doi.org/10.1145/2872427.2883081
- **Label field used:** `success` (utterance metadata) — binary indicator of
  whether a reply earned a delta (∆) from the original poster, signifying that
  the OP acknowledged a change in their view.
- **Splits:** Group-aware train/val/test split (80/10/10) using
  `GroupShuffleSplit` on `conversation_id` to prevent data leakage across
  partitions. Cached to CSV after first extraction.

---

## 4. Pre-trained Model

**Longformer-base-4096**

- **Model identifier:** `allenai/longformer-base-4096`
- **Source:** Hugging Face Model Hub — https://huggingface.co/allenai/longformer-base-4096
- **Developed by:** Allen Institute for AI
- **Paper:**
  Beltagy, I., Peters, M. E., & Cohan, A. (2020).
  Longformer: The Long-Document Transformer.
  *arXiv preprint arXiv:2004.05150*.
  https://arxiv.org/abs/2004.05150
- **License:** Apache 2.0
- **Usage:** Loaded via `AutoModel.from_pretrained()` and fine-tuned end-to-end
  for binary sequence classification. The classification head (a single linear
  layer) was randomly initialised and trained jointly with the backbone.

---

## 5. Other Resources

**Baseline reference — TF-IDF + Logistic Regression**
The majority-class and TF-IDF + Logistic Regression baselines follow standard
practice for text classification benchmarking and are not derived from any
specific third-party implementation.

**Evaluation metrics**
All metrics (F1, precision, recall, ROC-AUC) are computed using scikit-learn's
`sklearn.metrics` module with default settings unless otherwise noted in the
code.