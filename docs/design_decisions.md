

DESIGN DECISION: Triple Encoding vs Separate Encoding
======================================================

Problem with separate encoding (previous approach):
  - Each argument was encoded independently and scored separately
  - The model could use padding length as a proxy for argument length
  - Winners are systematically shorter (mean 91 words) than losers
    (mean 161 words) in this corpus
  - Result: the model achieved 0.977 val accuracy after 1 warmup epoch
    by learning 'shorter sequence = more padding = winner'
  - Length heuristic alone achieves 0.695 accuracy — much of the
    'model performance' was just length detection

Solution: triple encoding
  - Both arguments are packed into one sequence: [OP | ARG_A | ARG_B]
  - Both arguments share the same padded sequence length — the model
    cannot use padding as a length signal
  - Task becomes binary classification: 'which argument position wins?'
  - Position randomisation during training prevents positional bias
  - CrossEntropyLoss replaces MarginRankingLoss

DESIGN DECISION: Longformer vs DebertaV3
======================================================
  - Originally, project was fine-tuning a DebertaV3 model
  - Longformer was ultimately chosen over DebertaV3 due to its longer effective
    context length (4096 tokens vs 1536)
  - In the CMV corpus, many argument pairs require both arguments to be
    fully represented without truncation to capture nuanced rebuttals
    and supporting evidence
  - DebertaV3's shorter context would force aggressive truncation of
    longer arguments, potentially losing critical information
  - Longformer's efficient attention mechanism (combining local windowed
    and global attention) handles long sequences effectively while
    maintaining computational feasibility
  - This allows the model to learn from complete arguments rather than
    artificially shortened versions