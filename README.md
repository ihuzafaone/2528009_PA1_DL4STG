# AI651 — Deep Learning for Space, Time and Graphs · Programming Assignment 1

Huzaifa Abid — Roll No. 25280009 — LUMS, Fall 2026

| Path | Contents |
|---|---|
| `PA1_Report_25280009.pdf` | the report (both tasks) |
| `report/` | LaTeX source (`PA1_Report_25280009.tex`, `task1.tex`, `task2.tex`) |
| `Figures/` | every numbered PDF figure included in the report (Task 1 Outputs 1.2–4.2, Task 2 Fig. T2-1…T2-6) |
| `Question_1/` | executed full-preset Task 1 notebook, `harness/`, `requirements.txt`, trained checkpoints, `results/design/` (published outputs) |
| `Question_2/` | Task 2 Autoformer code, study logs and the submitted forecast — see `Question_2/README.md` |
| `DL4STG-PA1.pdf` | assignment handout |

## Task 1

```bash
cd Question_1
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
PA1_PRESET=full jupyter lab   # run Assignment1.ipynb top to bottom
```

The notebook sets `PA1_PRESET=full`. With the checkpoints in `checkpoints/` it reloads the four trained
networks (no retraining); delete them to train from scratch (about 25 min on an RTX 4060).

## Task 2

```bash
cd Question_2
python q2_v3.py study && python q2_v3.py final 5 && python q2_v3_report.py
```

## Rebuilding the report

```bash
cd report && tectonic PA1_Report_25280009.tex   # or xelatex / pdflatex
```
