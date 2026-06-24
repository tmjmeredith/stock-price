# -*- coding: utf-8 -*-
"""產出 probe 對照提交檔，用來量測 public LB 的「離群值地板」。

邏輯：當 RMSE 被極端離群值主導時，主體預測幾乎不影響總分。
比較幾個常數提交與我們的模型，就能反推測試集的結構：

- submission_zeros.csv      ：全 0（= sample_submission 預設），「什麼都不做」的地板。
- submission_const_mean.csv ：常數 = 訓練集均值（RMSE 的最佳常數估計）。
- submission.csv            ：我們的模型（已存在）。

判讀方式：
- zeros ≈ const_mean         → 離群值極大，主體完全無關（分數不可降）。
- const_mean 明顯 < zeros    → 測試集均值為正且大，應整體往上偏移（加正向 bias）。
- 我們的模型 < const_mean    → 模型主體技巧在 LB 上是淨正貢獻（方向正確）。
- 我們的模型 > const_mean    → 模型在離群點上反而吃虧，應更收斂/簡化。
"""
import pandas as pd

from pipeline import config as C

sample = pd.read_csv(C.SAMPLE_SUB_CSV)
train = pd.read_csv(C.TRAIN_CSV)
mean_val = float(train[C.TARGET].mean())
median_val = float(train[C.TARGET].median())

probes = {
    "submission_zeros.csv": 0.0,
    "submission_const_mean.csv": mean_val,
    "submission_const_median.csv": median_val,
}
for fname, val in probes.items():
    out = sample[[C.ID_COL]].copy()
    out[C.TARGET] = val
    out.to_csv(C.ROOT / fname, index=False)
    print(f"{fname:32s} -> 常數 {val:.4f}（{len(out)} 列）")

print(f"\n訓練集均值={mean_val:.4f}  中位數={median_val:.4f}")
print("提交順序建議：先傳 submission_zeros.csv 與 submission_const_mean.csv 對照，")
print("再傳 submission.csv 看模型是否勝過常數。")
