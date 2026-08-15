# Monthly Rolling Retrain V1.0 — Work9

## Production mechanism
When month T is closed and actual data is available:

```text
refresh Dataset through T
-> rebuild origin-safe features at T
-> retrain direct H1/H2/H3 models on expanding history
-> forecast T+1, T+2, T+3
-> save immutable vintage
```

Examples:
- data through Mar -> forecast Apr/May/Jun;
- Apr closes -> retrain through Apr -> forecast May/Jun/Jul;
- May closes -> retrain through May -> forecast Jun/Jul/Aug.

A target month therefore has multiple vintages, e.g. Jun is H3 from Mar, H2 from Apr, H1 from May. Older vintages are never overwritten; forecast revision is auditable.

## Zero rule in monthly refresh
For a known Pair, if month T is closed and there is no sales row, actual demand for T is 0 M² and is included in the retraining history as an observed zero. Negative-only months remain unavailable as gross-demand targets.
