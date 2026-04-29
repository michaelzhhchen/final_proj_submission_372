# CMV Winning Argument Classifier
A fine-tuned Longformer model that predicts which of two arguments successfully changed someone's view on Reddit's r/ChangeMyView. Given an original post and two competing replies, the model outputs a binary judgment of which argument is more persuasive; this model is used in a web-application that can be used to compare the effectiveness of two different arguments and decide which one is more persuasive.

---

## What It Does

The overarching goal of this project was to create a model that could predict whether an argument was more or less persuasive to someone else. This project fine-tunes `allenai/longformer-base-4096` on the ConvoKit winning-args-corpus to classify persuasive argument pairs from Reddit's r/ChangeMyView (CMV). Rather than encoding each argument separately and comparing scores — an approach that trivially learns to prefer shorter arguments — this model uses **triple encoding**: the original post, argument A, and argument B are packed into a single 2048-token sequence, forcing the model to read both arguments in context before making a judgment. Winner and loser positions are randomly swapped during training to prevent positional bias. The model is evaluated against three baselines (random, shorter-wins heuristic, and TF-IDF + Logistic Regression) and assessed on length-matched pairs to verify that it is learning content-based signals rather than length shortcuts. This project also includes a simple web-app, which takes in two arguments to be compared, as well as an optional "original argument" to provide additional context for evaluating the argument's persuasiveness. 

---

## Quick Start

### Requirements
```bash
pip install convokit transformers torch scikit-learn matplotlib seaborn accelerate
```

### Running the Notebook
Much of the project is implemented as a single Colab notebook. Open `cmv_longformer_triple.ipynb` in Google Colab and run all cells in order. A GPU runtime is required (Runtime → Change runtime type → A100 with high RAM activated). Note: because of the size of the CMV corpus, running the notebook in full took approximately an hour and a half. 

On first run, the notebook will:
1. Download the ConvoKit winning-args-corpus automatically
2. Build and cache train/val/test splits to `data/splits/`
3. Run a learning rate search over `[5e-6, 1e-5, 2e-5]`
4. Train the full model with the best LR for up to 8 epochs
5. Save the best checkpoint to `checkpoints/` and `export/`

On subsequent runs, cached splits are loaded automatically and training resumes from scratch unless a checkpoint is manually restored.

## Video Links

| Video | Link |
|-------|------|
| Demo Walkthrough | *[https://drive.google.com/file/d/1RQUQ6Pfvi3dkDfN90tMtOOK4XgZAqtfR/view?usp=sharing]* |
| Technical Walkthrough | *[https://drive.google.com/file/d/12rvqWCV3dwlF3Vi4q5vVo0_t4vSVuRNE/view?usp=sharing]* |

---

## Evaluation

### Baselines vs. Model

| Model | Accuracy |
|-------|----------|
| Random | 0.5000 |
| Shorter-wins heuristic | ~0.695 |
| TF-IDF + Logistic Regression | 0.9396 |
| **Longformer Triple-Encoding (ours)** | **0.9753** |

### Length-Matched Pairs
To verify the model is not exploiting length as a signal, accuracy was measured on the 43 test pairs where winner and loser length are within 20% of each other (a subset where the shorter-wins heuristic performs at chance):

**Accuracy on length-matched pairs: 97.7%**

### Accuracy by Winner Length Bucket

| Winner Length | Accuracy |
|---------------|----------|
| 0–50w | 98.7% |
| 51–100w | 98.4% |
| 101–200w | 100.0% |
| 201–400w | 89.5% |
| 400+w | 50.0% |

Performance degrades on very long winning arguments (400+ words), likely due to truncation within the 2048-token context window. See `error_analysis.md` for a full discussion of failure cases.

### Dataset
- **Source:** ConvoKit winning-args-corpus
- **Split:** 80% train / 10% val / 10% test (conversation-level)
- **Total tokens:** 2,618,981 (verified via Longformer tokenizer)