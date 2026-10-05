"""Task 2 pipeline v3 (roll no 25280009): features, validation folds, studies, final forecast.

    python q2_v3.py study    # multi-seed studies on two chronological folds -> results_v3/
    python q2_v3.py final    # trains the chosen config on all history, writes the 168 forecasts

Validation (chronological, no leakage). The hidden week is the last 168 rows of a 43,824-row
hourly series (5 years), i.e. a late-December week. Two folds:
  winter : train on targets ending before row 33600, score origins in rows 33600..36480
           (Nov-Feb of the previous winter - same season as the hidden week)
  recent : train on targets ending before row N - 16*168, score the last 16 weeks
Origins every 24 steps, all 168 leads scored, metrics on the raw scale.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from q2_v3_model import Autoformer, n_params

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_v3"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H = 168

y_raw = pd.read_csv(HERE / "Data/student_train.csv").value.to_numpy(np.float64)
EXT = pd.read_csv(HERE / "Data/optional_external_data.csv")
N, T = len(y_raw), len(EXT)
assert T == N + H
FOLDS = {"winter": (33600, 36480), "recent": (N - 16 * H, N)}

BASE = dict(feats="full", horizon_cov=True, trend="cov", target="raw", seq_len=96, d=16,
            k=25, factor=2, dr=0.1, epochs=4, lr=3e-3, bs=256)


# ------------------------------------------------------------------ features
def features(cfg, fit_end):
    """[T, M] covariates. Everything backward-looking in time; scaled on rows < fit_end."""
    e = EXT
    flags = e[["feature_G", "feature_H", "feature_I", "feature_J"]].to_numpy(float)
    cols, scale = [], []
    t = np.arange(1, T + 1)
    tod = [np.sin(2 * np.pi * t / 24), np.cos(2 * np.pi * t / 24)]
    doy = [np.sin(2 * np.pi * t / 8766), np.cos(2 * np.pi * t / 8766)]
    if cfg["feats"] != "none":
        A, B, C = (e[c].to_numpy(float) for c in ("feature_A", "feature_B", "feature_C"))
        Dc = e.feature_D.to_numpy(float)
        cols += [A, B, C, np.log1p(Dc), np.log1p(e.feature_E.to_numpy(float)),
                 np.log1p(e.feature_F.to_numpy(float))]
        scale += [True] * 6
        cols += list(flags.T)
        scale += [False] * 4
        if cfg["feats"] == "full":
            spread = B - A  # dew-point depression: small = humid air
            # hourly wind speed recovered from the cumulative column (restarts on direction change)
            direction = flags.argmax(1)
            w = Dc.copy()
            cont = (direction[1:] == direction[:-1]) & (Dc[1:] >= Dc[:-1])
            w[1:] = np.where(cont, Dc[1:] - Dc[:-1], Dc[1:])
            base = [w * flags[:, j] for j in range(4)] + [spread, flags[:, 3]]
            for hl in (6, 24, 72):  # exponential memory: pollution builds up / is flushed gradually
                for s in base:
                    cols.append(pd.Series(s).ewm(halflife=hl).mean().to_numpy())
                    scale.append(True)
            cols += [spread, C - np.r_[np.full(24, C[0]), C[:-24]]]
            scale += [True, True]
    cols += tod + doy
    scale += [False] * 4
    X = np.column_stack(cols).astype(np.float64)
    for j, s in enumerate(scale):
        if s:
            X[:, j] = (X[:, j] - X[:fit_end, j].mean()) / (X[:fit_end, j].std() + 1e-8)
    n_ext = X.shape[1] - 4  # columns that come from the optional file
    return torch.tensor(X, dtype=torch.float32, device=dev), n_ext


def target(cfg, fit_end):
    v = np.log1p(y_raw) if cfg["target"] == "log" else y_raw
    mu, sd = v[:fit_end].mean(), v[:fit_end].std()
    z = torch.tensor((v - mu) / sd, dtype=torch.float32, device=dev)

    def restore(p):
        p = p * sd + mu
        return np.clip(np.expm1(p) if cfg["target"] == "log" else p, 0, None)
    return z, restore


def inputs(cfg, z, X, n_ext, o):
    L, lab = cfg["seq_len"], cfg["seq_len"] // 2
    ci = o[:, None] + torch.arange(-L, 0, device=dev)
    di = o[:, None] + torch.arange(-lab, H, device=dev)
    cd = X[di]
    if not cfg["horizon_cov"]:  # ablation: optional data known only up to the origin
        cd = cd.clone()
        cd[:, lab:, :n_ext] = 0
    return z[ci].unsqueeze(-1), X[ci], cd


# ------------------------------------------------------------------ train / predict
def fit(cfg, seed, train_end):
    torch.manual_seed(seed)
    np.random.seed(seed)
    z, restore = target(cfg, train_end)
    X, n_ext = features(cfg, train_end)
    m = Autoformer(X.shape[1], cfg["seq_len"], cfg["seq_len"] // 2, H, d=cfg["d"], k=cfg["k"],
                   factor=cfg["factor"], dr=cfg["dr"], trend=cfg["trend"]).to(dev)
    orig = torch.arange(cfg["seq_len"], train_end - H + 1, device=dev)
    steps = cfg["epochs"] * math.ceil(len(orig) / cfg["bs"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-3)
    warm = max(1, int(0.1 * steps))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    hz = torch.arange(H, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(cfg["epochs"]):
        m.train()
        for idx in torch.randperm(len(orig), generator=g).split(cfg["bs"]):
            o = orig[idx.to(dev)]
            loss = F.mse_loss(m(*inputs(cfg, z, X, n_ext, o)), z[o[:, None] + hz])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
    m.eval()
    return m, (z, X, n_ext, restore)


@torch.no_grad()
def predict(cfg, m, ctx, origins):
    z, X, n_ext, restore = ctx
    o = torch.as_tensor(origins, device=dev)
    out = [m(*inputs(cfg, z, X, n_ext, o[i:i + 256])) for i in range(0, len(o), 256)]
    return restore(torch.cat(out).cpu().numpy().astype(np.float64))


def metrics(p, y):
    e = p - y
    tot = np.abs(p) + np.abs(y)
    sm = np.divide(2 * np.abs(e), tot, out=np.zeros_like(tot), where=tot > 0)
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()), smape=float(100 * sm.mean()))


def fold_origins(name):
    a, b = FOLDS[name]
    o = np.arange(a, b - H + 1, 24)
    return o, np.stack([y_raw[i:i + H] for i in o])


def members(cfg):
    return [dict(cfg, target="raw"), dict(cfg, target="log")] if cfg["target"] == "raw+log" else [cfg]


CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)


def member_forecast(mem, seed, fold):
    """Forecasts of one single-target model on one fold; cached on disk by its settings."""
    key = json.dumps({**mem, "seed": seed, "fold": fold}, sort_keys=True)
    f = CACHE / "index.json"
    index = json.loads(f.read_text()) if f.exists() else {}
    if key in index and Path(index[key]).exists():
        return np.load(index[key])
    m, ctx = fit(mem, seed, FOLDS[fold][0])
    p = predict(mem, m, ctx, fold_origins(fold)[0])
    path = CACHE / f"run{len(index):04d}.npy"
    np.save(path, p)
    index[key] = str(path)
    f.write_text(json.dumps(index))
    return p


def fold_forecast(cfg, seed, fold):
    return np.mean([member_forecast(mem, seed, fold) for mem in members(cfg)], 0)


def evaluate(cfg, seeds):
    """Per-seed RMSE (both folds pooled) mean/std, and RMSE of the seed average."""
    per_seed, avg = [], {}
    stacks = {f: [] for f in FOLDS}
    for s in seeds:
        errs = []
        for f in FOLDS:
            p = fold_forecast(cfg, s, f)
            stacks[f].append(p)
            errs.append(p - fold_origins(f)[1])
        per_seed.append(float(np.sqrt(np.mean(np.concatenate([e.ravel() for e in errs]) ** 2))))
    res = dict(rmse_mean=float(np.mean(per_seed)), rmse_std=float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0)
    pooled_p, pooled_y = [], []
    for f in FOLDS:
        p, y = np.mean(stacks[f], 0), fold_origins(f)[1]
        res[f"{f}_ens"] = metrics(p, y)
        pooled_p.append(p.ravel()); pooled_y.append(y.ravel())
    res["ens"] = metrics(np.concatenate(pooled_p), np.concatenate(pooled_y))
    return res


def baselines():
    rows = {}
    for f in FOLDS:
        o, y = fold_origins(f)
        rows[f] = {
            "train mean": metrics(np.full_like(y, y_raw[:FOLDS[f][0]].mean()), y),
            "last value": metrics(np.repeat(y_raw[o - 1][:, None], H, 1), y),
            "mean of last 168": metrics(np.stack([np.full(H, y_raw[i - H:i].mean()) for i in o]), y),
        }
    return rows


# ------------------------------------------------------------------ studies
def study():
    log = OUT / "study_log.json"
    rec = json.loads(log.read_text()) if log.exists() else {}
    rec["baselines"] = baselines()
    print(json.dumps(rec["baselines"], indent=1), flush=True)
    seeds = [0, 1, 2]
    cfg = dict(BASE)

    def run(name, options):
        nonlocal cfg
        rows = {}
        for label, change in options:
            t0 = time.time()
            c = dict(cfg, **change)
            r = evaluate(c, seeds)
            rows[label] = r
            print(f"[{name}] {label:28s} per-seed {r['rmse_mean']:6.2f} +- {r['rmse_std']:5.2f} | "
                  f"3-seed avg {r['ens']['rmse']:6.2f} (winter {r['winter_ens']['rmse']:6.2f}, "
                  f"recent {r['recent_ens']['rmse']:6.2f}) MAE {r['ens']['mae']:5.1f} "
                  f"sMAPE {r['ens']['smape']:4.1f}  [{time.time() - t0:.0f}s]", flush=True)
        best = min(rows, key=lambda k: rows[k]["rmse_mean"])
        cfg = dict(cfg, **dict(options)[best])
        print(f"[{name}] -> {best}\n", flush=True)
        rec[name] = dict(rows=rows, chosen=best, cfg=dict(cfg))
        log.write_text(json.dumps(rec, indent=1))

    run("S1 optional data (ablation)", [
        ("none", dict(feats="none")),
        ("raw file, past only", dict(feats="raw", horizon_cov=False)),
        ("raw file, past+future", dict(feats="raw")),
        ("engineered, past+future", dict(feats="full"))])
    run("S2 trend start", [("context mean (paper)", dict(trend="mean")),
                           ("mean + linear(cov)", dict(trend="cov"))])
    run("S3 target scale", [("raw", dict(target="raw")), ("log", dict(target="log")),
                            ("raw+log average", dict(target="raw+log"))])
    run("S4 epochs", [(f"{e} epochs", dict(epochs=e)) for e in (2, 4, 6)])
    run("S5 epochs (short end)", [(f"{e} epochs", dict(epochs=e)) for e in (1, 2, 3)])
    run("S6 width", [(f"d={d}", dict(d=d)) for d in (8, 16, 32)])
    run("S7 dropout", [(f"dropout {r}", dict(dr=r)) for r in (0.1, 0.3)])
    rec["final_cfg"] = cfg
    log.write_text(json.dumps(rec, indent=1))
    print("FINAL CONFIG:", cfg)


# ------------------------------------------------------------------ final forecast
def final(n_seeds=5):
    rec = json.loads((OUT / "study_log.json").read_text())
    cfg = rec["final_cfg"]
    preds, P, E = [], 0, 0
    for s in range(n_seeds):
        for mem in members(cfg):
            t0 = time.time()
            m, ctx = fit(mem, 1000 + s, N)  # refit on the whole history
            preds.append(predict(mem, m, ctx, [N])[0])
            P += n_params(m)
            E += mem["epochs"]
            print(f"seed {s} {mem['target']}: P={n_params(m)} mean forecast {preds[-1].mean():.1f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    fc = np.mean(preds, 0)
    assert fc.shape == (H,) and np.isfinite(fc).all() and (fc >= 0).all()
    line = ",".join(f"{v:.2f}" for v in fc)
    (OUT / "submission.txt").write_text(f"Params: {P}\nEpochs: {E}\nPredictions: {line}\n")
    pd.DataFrame({"time_idx": np.arange(N + 1, N + H + 1), "value": fc}).to_csv(OUT / "submission.csv", index=False)
    json.dump(dict(cfg=cfg, seeds=n_seeds, P=P, E=E, models=len(preds)), open(OUT / "final_info.json", "w"), indent=1)
    print(f"\nP = {P}   E = {E}\nPREDICTIONS:\n{line}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "study":
        study()
    elif cmd == "final":
        final(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif cmd == "speed":
        t0 = time.time(); m, _ = fit(dict(BASE, epochs=1), 0, FOLDS["recent"][0])
        print("1 epoch", time.time() - t0, "s; P =", n_params(m))
    else:
        raise SystemExit(__doc__)
