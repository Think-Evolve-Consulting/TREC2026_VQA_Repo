# Does the ranking signal predict correctness?

Generated 2026-08-15T16:16:43+00:00 from cached data; no inference.

> Re-ranking cannot change mean METEOR or mean BERTScore -- eval.vqa.ag.py averages over all ten answers, which is order-independent. Only NDCG responds to order.

## Mean METEOR by emitted rank

| Run | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r1 − r10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4 | 0.180 | 0.164 | 0.139 | 0.137 | 0.145 | 0.131 | 0.154 | 0.142 | 0.146 | 0.147 | +0.033 |
| qwen3vl | 0.077 | 0.060 | 0.052 | 0.066 | 0.052 | 0.054 | 0.046 | 0.041 | 0.047 | 0.040 | +0.037 |
| cosmos | 0.062 | 0.043 | 0.034 | 0.024 | 0.028 | 0.031 | 0.034 | 0.021 | 0.024 | 0.025 | +0.037 |

## Spearman correlation with per-answer METEOR

| Run | Signal | Pooled | Within-query mean | Queries |
|---|---|---|---|---|
| gemma4 | rank | 0.0232 | 0.106 | 140 |
| gemma4 | score | 0.0317 | 0.0974 | 139 |
| gemma4 | votes | -0.073 | -0.0147 | 126 |
| qwen3vl | rank | 0.0734 | 0.1338 | 125 |
| cosmos | rank | 0.0744 | 0.0873 | 113 |

## Signal resolution

| Run | Distinct scores of 10 | All ten tied | Answers at ≤1 vote |
|---|---|---|---|
| gemma4 | 5.29 | 1 | 69% |
