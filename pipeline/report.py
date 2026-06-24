# -*- coding: utf-8 -*-
"""診斷報告：整體 / 尾部 / 主體 RMSE、分年、分產業、相關性，輸出 oof_report.md。"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .cv import rmse


def tail_bulk_rmse(y, pred, top_k=50):
    """拆解 RMSE 來源：尾部（|y| 最大的 top_k 列）vs 主體（其餘）。"""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    order = np.argsort(-np.abs(y))
    tail_idx = order[:top_k]
    bulk_idx = order[top_k:]
    return {
        "overall": rmse(y, pred),
        f"tail_top{top_k}": rmse(y[tail_idx], pred[tail_idx]),
        "bulk": rmse(y[bulk_idx], pred[bulk_idx]),
    }


def segment_rmse(df, y, pred, by):
    """分組 RMSE（依年度或產業）。"""
    tmp = pd.DataFrame({"seg": df[by].values, "y": y, "p": pred})
    out = {}
    for seg, g in tmp.groupby("seg"):
        out[seg] = rmse(g["y"].values, g["p"].values)
    return out


def build_report(df, y, per_model_oof, blend_oof, final_oof, baselines,
                 weights, post_history, wf_results, model_params, path=None):
    """組裝並寫出 markdown 診斷報告。"""
    path = path or C.OOF_REPORT_MD
    L = []
    L.append("# 股票一年期報酬率預測 — OOF 診斷報告\n")
    L.append(f"- 訓練樣本數：{len(df):,}　特徵數：依 features.py 產生\n")

    # 基準線
    L.append("\n## 1. Naive 基準線（模型須勝過）\n")
    for k, v in baselines.items():
        L.append(f"- {k}: RMSE = {v:.4f}")

    # 各模型 OOF
    L.append("\n## 2. 各模型 OOF RMSE\n")
    for k, oof in per_model_oof.items():
        L.append(f"- {k}: RMSE = {rmse(y, oof):.4f}　(集成權重 {weights.get(k, 0):.3f})")
    L.append(f"- **集成 blend**: RMSE = {rmse(y, blend_oof):.4f}")

    # 後處理逐步
    L.append("\n## 3. 後處理逐步 OOF RMSE（最高槓桿）\n")
    h = post_history
    L.append(f"- 集成後: {h['blend']:.4f}")
    L.append(f"- 偏誤校正後 (a={h['a']:.4f}, b={h['b']:.4f}): {h['after_calibration']:.4f}")
    L.append(f"- 向均值收縮後 (alpha={h['alpha']:.3f}): {h['after_shrinkage']:.4f}")
    L.append(f"- 最佳裁剪後 (bounds={h['clip_bounds']}): {h['after_clip']:.4f}")
    L.append(f"- **最終 OOF RMSE = {rmse(y, final_oof):.4f}**")

    # RMSE 來源拆解
    L.append("\n## 4. RMSE 來源拆解（尾部 vs 主體）\n")
    for tag, pred in [("集成後", blend_oof), ("最終", final_oof)]:
        tb = tail_bulk_rmse(y, pred, top_k=50)
        L.append(f"- {tag}: 整體 {tb['overall']:.2f}｜尾部top50 {tb['tail_top50']:.2f}｜主體 {tb['bulk']:.2f}")

    # 輔助指標
    L.append("\n## 5. 輔助指標（最終預測）\n")
    mae = float(np.mean(np.abs(y - final_oof)))
    sp = spearmanr(y, final_oof).correlation
    L.append(f"- MAE = {mae:.4f}")
    L.append(f"- Spearman（排序相關）= {sp:.4f}")

    # 分年 / 分產業
    L.append("\n## 6. 分年 OOF RMSE（最終）\n")
    for seg, v in sorted(segment_rmse(df, y, final_oof, C.TIME_COL).items()):
        L.append(f"- {seg}: {v:.4f}")
    L.append("\n## 7. 分產業 OOF RMSE（最終）\n")
    for seg, v in sorted(segment_rmse(df, y, final_oof, C.CATEGORICAL[0]).items(),
                         key=lambda x: (np.isnan(x[0]) if isinstance(x[0], float) else False, x[0])):
        L.append(f"- sector {seg}: {v:.4f}")

    # walk-forward 時間穩健
    L.append("\n## 8. Walk-forward 分年診斷（時間穩健度）\n")
    for year, r in wf_results.items():
        L.append(f"- 驗證 {year}（集成後 blend）: RMSE = {r:.4f}")

    # 最佳參數
    L.append("\n## 9. Optuna 最佳參數\n")
    for mk, params in model_params.items():
        L.append(f"- **{mk}**: `{params}`")

    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
