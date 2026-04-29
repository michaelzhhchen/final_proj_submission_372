# Error Analysis

The model achieves **97.5% overall accuracy** (355/364) and **97.7% on length-matched pairs**, demonstrating that it has genuinely learned content-based signals rather than relying on length as a shortcut. However, the 9 misclassified cases reveal interpretable patterns.

---

## Performance by Winner Length

The clearest failure pattern is on **long winning arguments**. Accuracy drops sharply for winners over 200 words — from near-perfect performance in shorter buckets down to **50% (chance level)** for arguments over 400 words. This likely occurs because the model must allocate its 2048-token budget across the OP, winner, and loser simultaneously. When the winning argument is very long, it may be truncated, causing the model to lose the most persuasive portion of the text.

| Winner Length Bucket | Total Pairs | Correct | Accuracy |
|----------------------|-------------|---------|----------|
| 0–50w                | 155         | 153     | 98.7%    |
| 51–100w              | 122         | 120     | 98.4%    |
| 101–200w             | 62          | 62      | 100.0%   |
| 201–400w             | 19          | 17      | 89.5%    |
| 400+w                | 6           | 3       | 50.0%    |
