# Task 2 — Leaderboard Challenge (Roll No. 25280009)

Submitted model: **encoder–decoder Autoformer v3** (`q2_v3_model.py`), built by `q2_v3.py`.
Final submission: RMSE 68.34, MAE 51.54, sMAPE 61.91 %, score 68.341, P = 45,795, E = 10.

## Environment

| Item | Value |
|---|---|
| OS | Windows + WSL2 (Linux 5.15) |
| Python | 3.11.17 (conda env `pa1`) |
| PyTorch | 2.6.0+cu124 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| NumPy / pandas / matplotlib | 2.4.6 / 3.0.6 / 3.x |

## Reproduce

```bash
cd Question_2
python q2_v3.py study      # 7 multi-seed studies on two chronological folds -> results_v3/study_log.json
python q2_v3.py final 5    # 5 seeds refit on full history, 2 epochs each -> results_v3/submission.{csv,txt}
python q2_v3_report.py     # report figures + results_v3/report_numbers.json (reuses the study cache)
```

`results_v3/submission.txt` contains the declared `Params: 45795`, `Epochs: 10`, and the 168
comma-separated forecasts for `time_idx` 43657 … 43824 in chronological order (the exact string
pasted into the leaderboard). `P` is counted with `sum(p.numel() for p in m.parameters() if p.requires_grad)`
summed over the 5 ensemble members (5 × 9,159); `E` = 5 members × 2 epochs (fixed schedule, no early stopping).

## Files

| File | Purpose |
|---|---|
| `q2_v3_model.py` | Autoformer: progressive series decomposition (moving average, kernel 25) + Auto-Correlation (FFT delay scores, top-⌊2 ln L⌋ delays, learned temperature, time-delay aggregation); no dot-product attention |
| `q2_v3.py` | features, two-fold chronological validation, studies S1–S7, final 5-seed forecast |
| `q2_v3_report.py` | figures T2-1 … T2-6 and the numbers quoted in the report |
| `q2_v3_study_log.txt`, `results_v3/study_log.json` | full study output (per-seed mean ± std, 3-seed average, per-fold RMSE/MAE/sMAPE) |
| `results_v3/final_info.json` | final config, seeds, P, E |
| `Data/` | the three files supplied with the assignment |

## Validation protocol (summary)

Two chronological folds, each trained only on rows before it: *winter* (rows 33,600–36,480) and
*recent* (last 16 weeks). Forecast origins every 24 steps, all 168 leads scored on the raw scale,
3 seeds per option; the option with the lowest mean per-seed pooled RMSE is chosen. See §2.2–2.6 of
the report for all results, including the with/without optional-data ablation (S1).

## AI use

Claude (Anthropic) via Cursor was used for code implementation, debugging and drafting; the full
prompt / output / edit record is in the report's appendix.
