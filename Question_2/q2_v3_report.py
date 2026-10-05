"""Figures and numbers for the Task 2 report, from the stored v3 runs (nothing is trained here).

    python q2_v3_report.py      -> results_v3/figures/*.pdf and results_v3/report_numbers.json
"""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import q2_v3 as q

FIG = q.OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
rec = json.loads((q.OUT / "study_log.json").read_text())
FINAL = rec["final_cfg"]
SEEDS = [0, 1, 2]
y = q.y_raw
numbers = {}


def cached(cfg, fold):
    return np.mean([q.fold_forecast(cfg, s, fold) for s in SEEDS], 0)


# ---- data facts
def acf(x, lag):
    x = x - x.mean()
    return float((x[:-lag] * x[lag:]).sum() / (x * x).sum())


numbers["data"] = dict(mean=float(y.mean()), std=float(y.std()), median=float(np.median(y)),
                       max=float(y.max()), p99=float(np.quantile(y, 0.99)),
                       acf={l: acf(y, l) for l in (1, 6, 24, 48, 168)})
ext = q.EXT
numbers["corr"] = {c: float(np.corrcoef(ext[c].to_numpy()[:q.N], y)[0, 1]) for c in ext.columns[1:]}
numbers["flags_in_horizon"] = {c: int(ext[c].iloc[q.N:].sum()) for c in ["feature_G", "feature_H", "feature_I", "feature_J"]}
numbers["mean_by_flag"] = {c: float(y[ext[c].to_numpy()[:q.N] == 1].mean()) for c in ["feature_G", "feature_H", "feature_I", "feature_J"]}

fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
ax[0].plot(np.arange(q.N - 8 * 168, q.N), y[-8 * 168:], lw=0.6)
ax[0].set_title("Last 8 weeks of the target"); ax[0].set_xlabel("time_idx")
lags = np.arange(1, 337)
ax[1].plot(lags, [acf(y, l) for l in lags]); ax[1].axvline(24, ls=":", c="k"); ax[1].axvline(168, ls=":", c="k")
ax[1].set_title("Autocorrelation"); ax[1].set_xlabel("lag")
prof = pd.Series(y).groupby(np.arange(q.N) % 24).mean()
ax[2].plot(prof.index, prof.values, marker=".")
ax[2].set_title("Mean by index mod 24"); ax[2].set_xlabel("index mod 24")
fig.tight_layout(); fig.savefig(FIG / "T2-1_data.pdf"); plt.close(fig)

# ---- ablation figure (S1)
s1 = rec["S1 optional data (ablation)"]["rows"]
labels = list(s1)
fig, ax = plt.subplots(figsize=(6, 2.6))
means = [s1[k]["rmse_mean"] for k in labels]; stds = [s1[k]["rmse_std"] for k in labels]
ax.barh(range(len(labels)), means, xerr=stds, color=["#999", "#bbb", "#6a9fd8", "#2b6cb0"])
for i, k in enumerate(labels):
    ax.text(means[i] + stds[i] + 2, i, f"{means[i]:.1f}", va="center")
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
base = rec["baselines"]
tm = np.mean([base[f]["train mean"]["rmse"] for f in base])
ax.axvline(tm, ls="--", c="r", lw=0.8); ax.text(tm + 1, 3.4, "training-mean baseline", color="r", fontsize=7)
ax.set_xlabel("validation RMSE (mean ± std over 3 seeds, both folds)")
fig.tight_layout(); fig.savefig(FIG / "T2-2_ablation.pdf"); plt.close(fig)

# ---- epochs figure
ep = {**rec["S4 epochs"]["rows"], **rec["S5 epochs (short end)"]["rows"]}
order = sorted(ep, key=lambda k: int(k.split()[0]))
fig, ax = plt.subplots(figsize=(4, 2.6))
xs = [int(k.split()[0]) for k in order]
ax.errorbar(xs, [ep[k]["rmse_mean"] for k in order], [ep[k]["rmse_std"] for k in order], marker="o", capsize=3, label="single seed")
ax.plot(xs, [ep[k]["ens"]["rmse"] for k in order], marker="s", label="3-seed average")
ax.set_xlabel("epochs"); ax.set_ylabel("validation RMSE"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "T2-3_epochs.pdf"); plt.close(fig)

# ---- validation forecasts, lead-time error and high values for the final config
none_cfg = dict(FINAL, feats="none")
fig, ax = plt.subplots(2, 1, figsize=(10, 4.2))
numbers["final_val"] = {}
lead = {}
for i, fold in enumerate(q.FOLDS):
    o, truth = q.fold_origins(fold)
    p = cached(FINAL, fold)
    numbers["final_val"][fold] = q.metrics(p, truth)
    lead[fold] = np.sqrt(((p - truth) ** 2).mean(0))
    # every 7th origin = non-overlapping weeks
    sel = np.arange(0, len(o), 7)[:4]
    t = np.concatenate([np.arange(o[j], o[j] + q.H) for j in sel])
    ax[i].plot(t, np.concatenate([truth[j] for j in sel]), lw=0.7, c="k", label="truth")
    ax[i].plot(t, np.concatenate([p[j] for j in sel]), lw=0.9, c="#2b6cb0", label="Autoformer v3 (3-seed avg)")
    ax[i].set_title(f"{fold} fold: four consecutive non-overlapping weeks"); ax[i].legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "T2-4_val_forecasts.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 2.6))
for fold in q.FOLDS:
    ax.plot(np.arange(1, q.H + 1), lead[fold], label=f"{fold} fold")
p_none = {f: cached(none_cfg, f) for f in q.FOLDS}
ax.plot(np.arange(1, q.H + 1), np.sqrt(np.mean([((p_none[f] - q.fold_origins(f)[1]) ** 2).mean(0) for f in q.FOLDS], 0)),
        ls="--", c="gray", label="no optional data (both folds)")
ax.set_xlabel("lead time (steps)"); ax.set_ylabel("RMSE"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "T2-5_lead_time.pdf"); plt.close(fig)
numbers["lead_rmse"] = {f: dict(first24=float(np.sqrt((lead[f][:24] ** 2).mean())),
                                last24=float(np.sqrt((lead[f][-24:] ** 2).mean()))) for f in q.FOLDS}

# share of squared error on the top 10% values
pp = np.concatenate([cached(FINAL, f).ravel() for f in q.FOLDS])
tt = np.concatenate([q.fold_origins(f)[1].ravel() for f in q.FOLDS])
hi = tt > np.quantile(tt, 0.9)
e = pp - tt
numbers["high_values"] = dict(threshold=float(np.quantile(tt, 0.9)),
                              share_sq_err_top10=float((e[hi] ** 2).sum() / (e ** 2).sum()),
                              mean_err_top10=float(e[hi].mean()), mean_err_rest=float(e[~hi].mean()))

# ---- final forecast
sub = pd.read_csv(q.OUT / "submission.csv")
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.plot(np.arange(q.N - 336 + 1, q.N + 1), y[-336:], c="k", lw=0.7, label="last two weeks of history")
ax.plot(sub.time_idx, sub.value, c="#2b6cb0", label="submitted forecast (5-seed average)")
ax2 = ax.twinx()
ax2.fill_between(sub.time_idx, 0, ext.feature_H.iloc[q.N:].to_numpy(), color="orange", alpha=0.2, step="mid",
                 label="feature_H = 1")
ax2.set_yticks([]); ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
ax.set_xlabel("time_idx")
fig.tight_layout(); fig.savefig(FIG / "T2-6_forecast.pdf"); plt.close(fig)

(q.OUT / "report_numbers.json").write_text(json.dumps(numbers, indent=1))
print(json.dumps(numbers, indent=1))
