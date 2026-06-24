# -*- coding: utf-8 -*-
"""用已保存的 artifacts，比較「GBDT-only」與「GBDT+MLP」集成的 OOF RMSE。"""
import numpy as np
from scipy.optimize import minimize
from pipeline import config as C

d = np.load(C.ARTIFACT_DIR / "preds.npz", allow_pickle=True)
y = d["y"]
models = [k[4:] for k in d.files if k.startswith("oof_")]
print("可用模型：", models)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def best_blend(keys):
    P = np.column_stack([d[f"oof_{k}"] for k in keys])
    m = P.shape[1]
    w0 = np.full(m, 1 / m)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0, 1)] * m
    res = minimize(lambda w: rmse(y, P @ w), w0, method="SLSQP",
                   bounds=bnds, constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None); w /= w.sum()
    return rmse(y, P @ w), dict(zip(keys, np.round(w, 3)))

for keys in [["lgbm", "xgb", "cat"], ["lgbm", "xgb", "cat", "mlp"]]:
    r, w = best_blend(keys)
    print(f"{'+'.join(keys):25s} blend OOF RMSE = {r:.4f}  weights={w}")
