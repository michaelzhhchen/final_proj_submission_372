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

Inference change:
  - Single-post scoring is no longer direct — use compare_arguments()
    to rank two candidates head-to-head, or rank_arguments() for
    round-robin ranking of multiple candidates
  - This is actually more aligned with real use cases (debate prep,
    draft comparison) where you always have two options to compare

Key evaluation metric:
  - Overall accuracy vs random (0.50) and shorter-wins (0.695)
  - Accuracy on length-matched pairs (within 20% length ratio) is the
    cleanest signal — shorter-wins scores 0.50 there by construction,
    so any accuracy above 0.50 on those pairs is genuine content learning
