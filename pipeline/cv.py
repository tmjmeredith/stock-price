# -*- coding: utf-8 -*-
"""驗證設計：GroupKFold(by ticker) 主 CV、walk-forward 分年診斷、naive 基準線。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C


def rmse(y_true, y_pred):
    """原始尺度 RMSE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def group_folds(df):
    """主 CV：GroupKFold by ticker，回傳 [(train_idx, valid_idx), ...]。

    驗證折的 ticker 在訓練折未出現 → 貼近測試集為「新 ticker」的情境，
    同時每列恰被預測一次，產生全覆蓋 OOF（後處理/集成必需）。
    """
    gkf = GroupKFold(n_splits=C.N_FOLDS)
    groups = df[C.GROUP_COL].values
    return list(gkf.split(df, groups=groups))


def walk_forward_folds(df):
    """診斷用：擴張視窗的 walk-forward 分年切分。

    train<=Y -> val (Y+1)，檢查時間/regime 穩健度。
    回傳 [(year, train_idx, valid_idx), ...]。
    """
    years = sorted(df[C.TIME_COL].unique())
    folds = []
    for i in range(1, len(years)):
        val_year = years[i]
        train_mask = df[C.TIME_COL] < val_year
        valid_mask = df[C.TIME_COL] == val_year
        folds.append((
            int(val_year),
            np.where(train_mask.values)[0],
            np.where(valid_mask.values)[0],
        ))
    return folds


def naive_baselines(df):
    """naive 基準線 RMSE：全域均值、sector x year 群組均值。

    模型必須勝過這些；勝幅小代表訊號極弱，後處理應更收縮。
    """
    y = df[C.TARGET].values
    res = {}

    # 全域均值（RMSE 的最佳常數預測）
    res["global_mean"] = rmse(y, np.full_like(y, y.mean(), dtype=float))

    # sector x year 群組均值（leave-out 近似：直接用群組均值，僅供參考量級）
    grp = df.groupby([C.CATEGORICAL[0], C.TIME_COL])[C.TARGET].transform("mean")
    grp = grp.fillna(y.mean()).values
    res["sector_year_mean"] = rmse(y, grp)
    return res
