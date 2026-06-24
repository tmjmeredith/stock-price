# -*- coding: utf-8 -*-
"""後處理（最高槓桿）：集成權重、偏誤校正、最佳裁剪、向均值收縮。

所有參數一律在 OOF（預測+真實值）上決定，再原封不動套用到測試集。
最終組裝順序：集成 → 偏誤校正 → 向均值收縮 → 最佳裁剪（皆貪婪地降低 OOF RMSE）。
"""
import numpy as np
from scipy.optimize import minimize

from .cv import rmse


# ---------------------------------------------------------------------------
# 1. 集成權重（非負、和為 1，最小化 OOF RMSE）
# ---------------------------------------------------------------------------
def optimize_ensemble_weights(oof_dict, y):
    """對多模型 OOF 求非負且和為 1 的最佳混合權重。"""
    keys = list(oof_dict.keys())
    P = np.column_stack([oof_dict[k] for k in keys])  # (n, m)
    m = P.shape[1]

    def obj(w):
        return rmse(y, P @ w)

    w0 = np.full(m, 1.0 / m)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * m
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-10})
    w = res.x if res.success else w0
    w = np.clip(w, 0, None)
    w = w / w.sum()
    weights = {k: float(wi) for k, wi in zip(keys, w)}
    blend = P @ w
    return weights, blend


def apply_weights(test_dict, weights):
    """將集成權重套用到測試集各模型預測。"""
    return sum(weights[k] * np.asarray(test_dict[k]) for k in weights)


# ---------------------------------------------------------------------------
# 2. 偏誤校正（線性 a*x + b，在 OOF 上最小平方）
# ---------------------------------------------------------------------------
def fit_bias_calibration(pred, y):
    """擬合線性校正係數 (a, b)；若無法改善則回傳 (1, 0)。"""
    a, b = np.polyfit(pred, y, 1)
    base = rmse(y, pred)
    cal = rmse(y, a * pred + b)
    if cal < base:
        return float(a), float(b)
    return 1.0, 0.0


def apply_calibration(pred, a, b):
    return a * np.asarray(pred) + b


# ---------------------------------------------------------------------------
# 3. 向均值收縮：pred_final = alpha*pred + (1-alpha)*baseline
# ---------------------------------------------------------------------------
def search_shrinkage(pred, y, baseline, alphas=None):
    """在 OOF 上搜尋最佳收縮係數 alpha。"""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 101)
    baseline = np.asarray(baseline, dtype=float)
    best_a, best_r = 1.0, rmse(y, pred)
    for a in alphas:
        r = rmse(y, a * pred + (1 - a) * baseline)
        if r < best_r:
            best_r, best_a = r, a
    return float(best_a), float(best_r)


def apply_shrinkage(pred, baseline, alpha):
    return alpha * np.asarray(pred) + (1 - alpha) * np.asarray(baseline, dtype=float)


# ---------------------------------------------------------------------------
# 4. 最佳裁剪：在 OOF 上格點搜尋上 / 下界
# ---------------------------------------------------------------------------
def search_clip(pred, y, lowers=None, uppers=None):
    """搜尋使 OOF RMSE 最低的 (lower, upper) 裁剪界。"""
    if uppers is None:
        uppers = [40, 50, 60, 70, 80, 100, 120, 150, 200, 300, 500, 1000, np.inf]
    if lowers is None:
        lowers = [-100, -95, -90, -80, -70, -60, -50, -np.inf]
    best = (-np.inf, np.inf)
    best_r = rmse(y, pred)
    for lo in lowers:
        for hi in uppers:
            if lo >= hi:
                continue
            r = rmse(y, np.clip(pred, lo, hi))
            if r < best_r:
                best_r, best = r, (float(lo), float(hi))
    return best, float(best_r)


def apply_clip(pred, bounds):
    lo, hi = bounds
    return np.clip(np.asarray(pred), lo, hi)


# ---------------------------------------------------------------------------
# 串接：在 OOF 上鎖定全部參數，回傳一個可套用到測試集的 transform 物件
# ---------------------------------------------------------------------------
class PostProcessor:
    """封裝集成後的後處理鏈（校正→收縮→裁剪），參數全由 OOF 鎖定。"""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.alpha = 1.0
        self.clip_bounds = (-np.inf, np.inf)
        self.history = {}

    def fit(self, blend_oof, y, baseline_oof):
        r0 = rmse(y, blend_oof)
        # 偏誤校正
        self.a, self.b = fit_bias_calibration(blend_oof, y)
        p = apply_calibration(blend_oof, self.a, self.b)
        r1 = rmse(y, p)
        # 向均值收縮
        self.alpha, _ = search_shrinkage(p, y, baseline_oof)
        p = apply_shrinkage(p, baseline_oof, self.alpha)
        r2 = rmse(y, p)
        # 最佳裁剪
        self.clip_bounds, _ = search_clip(p, y)
        p = apply_clip(p, self.clip_bounds)
        r3 = rmse(y, p)
        self.history = {
            "blend": r0, "after_calibration": r1,
            "after_shrinkage": r2, "after_clip": r3,
            "a": self.a, "b": self.b, "alpha": self.alpha,
            "clip_bounds": self.clip_bounds,
        }
        return p

    def transform(self, blend, baseline):
        p = apply_calibration(blend, self.a, self.b)
        p = apply_shrinkage(p, baseline, self.alpha)
        p = apply_clip(p, self.clip_bounds)
        return p
