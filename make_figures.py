# -*- coding: utf-8 -*-
"""產生關鍵診斷圖表，存到 figures/，供 README 引用（用 artifacts 快取，免重訓）。

圖表（軸標籤用英文以避免字型缺字）：
  fig_target_dist.png        目標分布（全域 log-y + 主體放大）
  fig_tail_concentration.png 尾部平方誤差累積占比
  fig_pred_vs_actual.png     OOF 預測 vs 實際（主體）
  fig_residual_vs_pred.png   殘差 vs 預測（主體）
  fig_rmse_by_year.png       分年 OOF RMSE
  fig_rmse_by_sector.png     分產業 OOF RMSE
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import config as C
from pipeline import postprocess as PP
from pipeline.cv import rmse

FIG = C.ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---- 載入：GBDT-only blend → 後處理 final OOF ----
d = np.load(C.ARTIFACT_DIR / "preds.npz", allow_pickle=True)
y = d["y"]
w = {"lgbm": 0.098, "xgb": 0.655, "cat": 0.247}
blend = sum(w[k] * d[f"oof_{k}"] for k in w)
train = pd.read_csv(C.TRAIN_CSV)
glob = train[C.TARGET].mean()
sector_mean = train.groupby(C.CATEGORICAL[0])[C.TARGET].mean()
base_oof = train[C.CATEGORICAL[0]].map(sector_mean).fillna(glob).values
post = PP.PostProcessor()
final = post.fit(blend, y, base_oof)
resid = y - final
print(f"final OOF RMSE = {rmse(y, final):.4f}")

# ---- 1. 目標分布 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(y, bins=120, color="steelblue")
ax[0].set_yscale("log")
ax[0].set_title("Target distribution (full range, log-y)")
ax[0].set_xlabel("return_pct"); ax[0].set_ylabel("count (log)")
mask = (y >= -100) & (y <= 200)
ax[1].hist(y[mask], bins=80, color="seagreen")
ax[1].set_title("Target distribution (bulk, -100~200%)")
ax[1].set_xlabel("return_pct"); ax[1].set_ylabel("count")
fig.tight_layout(); fig.savefig(FIG / "fig_target_dist.png", dpi=120); plt.close(fig)

# ---- 2. 尾部平方誤差累積占比（相對均值的平方離差）----
sq = np.sort((y - y.mean()) ** 2)[::-1]
cum = np.cumsum(sq) / sq.sum()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(1, len(cum) + 1), cum, color="firebrick")
ax.set_xscale("log")
for k, c in [(10, "k"), (50, "gray")]:
    ax.axvline(k, ls="--", color=c, lw=1)
    ax.text(k, 0.05, f"top{k}: {cum[k-1]*100:.0f}%", rotation=90, va="bottom", fontsize=8)
ax.set_title("Tail concentration of squared deviation")
ax.set_xlabel("rank of |deviation| (log)"); ax.set_ylabel("cumulative share")
ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(FIG / "fig_tail_concentration.png", dpi=120); plt.close(fig)

# ---- 3. 預測 vs 實際（主體）----
fig, ax = plt.subplots(figsize=(5, 5))
m = (y >= -100) & (y <= 300)
ax.scatter(final[m], y[m], s=4, alpha=0.15, color="steelblue")
lims = [-100, 300]
ax.plot(lims, lims, "r--", lw=1, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title("OOF prediction vs actual (bulk)")
ax.set_xlabel("predicted return_pct"); ax.set_ylabel("actual return_pct")
ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig_pred_vs_actual.png", dpi=120); plt.close(fig)

# ---- 4. 殘差 vs 預測（主體）----
fig, ax = plt.subplots(figsize=(6, 4))
m = (final >= -60) & (final <= 120)
ax.scatter(final[m], resid[m], s=4, alpha=0.12, color="darkorange")
ax.axhline(0, color="k", lw=1)
ax.set_ylim(-250, 400)
ax.set_title("Residual vs prediction (bulk)")
ax.set_xlabel("predicted return_pct"); ax.set_ylabel("residual (actual - pred)")
fig.tight_layout(); fig.savefig(FIG / "fig_residual_vs_pred.png", dpi=120); plt.close(fig)

# ---- 5. 分年 OOF RMSE ----
yrs = sorted(train[C.TIME_COL].unique())
ry = [rmse(y[train[C.TIME_COL].values == yr], final[train[C.TIME_COL].values == yr]) for yr in yrs]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar([str(v) for v in yrs], ry, color="slateblue")
for i, v in enumerate(ry):
    ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_title("OOF RMSE by year"); ax.set_xlabel("start_year"); ax.set_ylabel("RMSE")
fig.tight_layout(); fig.savefig(FIG / "fig_rmse_by_year.png", dpi=120); plt.close(fig)

# ---- 6. 分產業 OOF RMSE ----
sec = train[C.CATEGORICAL[0]].values
secs = sorted([s for s in np.unique(sec) if not np.isnan(s)])
rs = [rmse(y[sec == s], final[sec == s]) for s in secs]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([f"{int(s)}" for s in secs], rs, color="teal")
ax.set_title("OOF RMSE by sector"); ax.set_xlabel("sector_code"); ax.set_ylabel("RMSE")
fig.tight_layout(); fig.savefig(FIG / "fig_rmse_by_sector.png", dpi=120); plt.close(fig)

print("已輸出 6 張圖到", FIG)
