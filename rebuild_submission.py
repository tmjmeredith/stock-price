# -*- coding: utf-8 -*-
"""從 artifacts/preds.npz 重建 submission.csv（不需重訓）。

用法：
    python rebuild_submission.py lgbm xgb cat        # GBDT-only（= 14753 版）
    python rebuild_submission.py lgbm xgb cat mlp     # 含 MLP
套用與 run.py 相同的後處理：集成權重 → 偏誤校正 → 收縮 → 裁剪（全由 OOF 鎖定）。
"""
import sys
import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import postprocess as PP
from pipeline.cv import rmse

keys = sys.argv[1:] or ["lgbm", "xgb", "cat"]

d = np.load(C.ARTIFACT_DIR / "preds.npz", allow_pickle=True)
y = d["y"]
test_id = d["test_id"]
oof = {k: d[f"oof_{k}"] for k in keys}
test = {k: d[f"test_{k}"] for k in keys}

# baseline：產業均值（測試集 2024 無標籤，沿用訓練集產業均值）
train = pd.read_csv(C.TRAIN_CSV)
test_df = pd.read_csv(C.TEST_CSV)
glob = train[C.TARGET].mean()
sector_mean = train.groupby(C.CATEGORICAL[0])[C.TARGET].mean()
base_oof = train[C.CATEGORICAL[0]].map(sector_mean).fillna(glob).values
base_test = test_df[C.CATEGORICAL[0]].map(sector_mean).fillna(glob).values

# 集成 → 後處理
weights, blend_oof = PP.optimize_ensemble_weights(oof, y)
blend_test = PP.apply_weights(test, weights)
post = PP.PostProcessor()
final_oof = post.fit(blend_oof, y, base_oof)
final_test = post.transform(blend_test, base_test)

print(f"模型={keys}")
print(f"集成權重={ {k: round(v,3) for k,v in weights.items()} }")
print(f"blend OOF={rmse(y, blend_oof):.4f}  最終 OOF={rmse(y, final_oof):.4f}")

final_test = np.nan_to_num(final_test, nan=float(y.mean()))
sample = pd.read_csv(C.SAMPLE_SUB_CSV)
sub = pd.DataFrame({C.ID_COL: test_id, C.TARGET: final_test})
sub = sample[[C.ID_COL]].merge(sub, on=C.ID_COL, how="left")
assert sub[C.TARGET].isna().sum() == 0 and len(sub) == len(sample)
sub.to_csv(C.SUBMISSION_CSV, index=False)
print(f"已寫出 {C.SUBMISSION_CSV}（{len(sub)} 列）")
