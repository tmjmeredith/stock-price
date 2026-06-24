# -*- coding: utf-8 -*-
"""資料載入與基本清理。

只做「不依賴標籤、不跨樣本」的安全清理：型別轉換、無限值轉 NaN。
特徵工程一律放到 features.py，避免訓練/測試處理不一致。
"""
import numpy as np
import pandas as pd

from . import config as C


def load_raw():
    """載入 train / test / sample_submission 原始資料。"""
    train = pd.read_csv(C.TRAIN_CSV)
    test = pd.read_csv(C.TEST_CSV)
    sample_sub = pd.read_csv(C.SAMPLE_SUB_CSV)
    return train, test, sample_sub


def basic_clean(df):
    """逐列、不跨樣本的安全清理。

    - 將 +/-inf 轉成 NaN（GBDT 原生吃 NaN，但 inf 會出問題）。
    - sector_code 轉為整數類別（保留 NaN）。
    """
    df = df.copy()
    # 數值欄的 inf -> NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def get_groups(df):
    """取出 CV 群組（ticker）。"""
    return df[C.GROUP_COL].values
