# -*- coding: utf-8 -*-
"""特徵工程：train / test 共用同一函式，嚴格可重現、無洩漏。

設計原則
--------
- 只用「測試集也能重現」的特徵：原始數值、缺失指示、橫斷面年內排名、
  產業相對 z-score、規模代理、衍生比率、sector_code 類別。
- 棄用 ticker（測試集匿名零重疊）、period_start/period_end（測試集無、且洩漏）、
  跨時間 panel lag（測試集無日期排序、單一年度，無法重現）。
- 所有「跨樣本」統計（排名、產業 z-score）一律**在各自 DataFrame 的 start_year 群組內**
  計算。測試集全為 2024 自成 cohort，與訓練集完全隔離，故無時間洩漏。
"""
import numpy as np
import pandas as pd

from . import config as C


def _safe_zscore(s: pd.Series) -> pd.Series:
    """群組內 z-score；std 為 0 或 NaN 時回傳 0。"""
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_features(df: pd.DataFrame):
    """由清理後的原始 DataFrame 產生特徵矩陣。

    回傳
    ----
    X : pd.DataFrame
        特徵矩陣（含類別欄 sector_code 以 category 型別表示）。
    feature_names : list[str]
        特徵欄名。
    cat_features : list[str]
        類別特徵欄名（供 LGBM/CatBoost 使用）。
    """
    df = df.copy()
    out = pd.DataFrame(index=df.index)

    year = df[C.TIME_COL]  # start_year，分年群組鍵

    # ---- 1. 原始數值特徵（保留 NaN，GBDT 原生處理） ----
    for col in C.RAW_NUMERIC:
        out[col] = df[col]

    # ---- 2. 缺失指示特徵 + 每列缺失總數 ----
    # 對所有原始數值欄建 is_missing_，保證 train/test 欄位完全一致（全 0 欄無害）。
    miss_block = df[C.RAW_NUMERIC].isna()
    for col in C.RAW_NUMERIC:
        out[f"is_missing_{col}"] = miss_block[col].astype("int8")
    out["n_missing"] = miss_block.sum(axis=1).astype("int16")

    # ---- 3. 橫斷面年內百分位排名（消除跨年水準差異，核心特徵） ----
    for col in C.RANK_FEATURES:
        out[f"{col}_rank"] = df.groupby(year)[col].rank(pct=True)

    # ---- 4. 產業相對 z-score（sector_code x start_year 內標準化） ----
    grp_keys = [df[C.CATEGORICAL[0]], year]
    for col in C.SECTOR_REL_FEATURES:
        out[f"{col}_secz"] = df.groupby(grp_keys)[col].transform(_safe_zscore)

    # ---- 5. 規模代理（尾部風險訊號） ----
    # 市值代理：以權益市值 / 盈餘×PE 估算（取絕對量級再 log1p）。
    mktcap_pb = (df["price_to_book"] * df["stockholders_equity"]).abs()
    mktcap_pe = (df["pe_ttm"] * df["net_income_ttm"]).abs()
    size_raw = {
        "size_revenue": df["revenue_ttm"].abs(),
        "size_assets": df["total_assets"].abs(),
        "size_mktcap_pb": mktcap_pb,
        "size_mktcap_pe": mktcap_pe,
    }
    for name, val in size_raw.items():
        logv = np.log1p(val)
        out[f"log_{name}"] = logv
        # 年內規模排名：小規模 → 尾部風險高
        out[f"{name}_rank"] = logv.groupby(year).rank(pct=True)

    # ---- 6. 衍生財務比率 ----
    def _ratio(num, den):
        return df[num] / df[den].replace(0, np.nan)

    out["ni_to_assets"] = _ratio("net_income_ttm", "total_assets")
    out["ltd_to_assets"] = _ratio("long_term_debt", "total_assets")
    out["goodwill_to_assets"] = _ratio("goodwill", "total_assets")
    out["inventory_to_curassets"] = _ratio("inventory", "current_assets")
    out["ca_to_cl"] = _ratio("current_assets", "current_liabilities")
    out["dilution_gap"] = df["eps_diluted"] - df["eps_basic"]
    out["equity_to_assets"] = _ratio("stockholders_equity", "total_assets")
    out["income_quality"] = _ratio("net_income_ttm", "income_before_tax")

    # ---- 7. 類別特徵 + 年度 ----
    # sector_code 以「字串」類別表示（XGBoost 的 enable_categorical 不接受浮點類別，
    # 字串為三套 GBDT 都接受的格式；NaN 保留為缺失）。
    out[C.CATEGORICAL[0]] = (
        df[C.CATEGORICAL[0]].round().astype("Int64").astype("string").astype("category")
    )
    out[C.TIME_COL] = df[C.TIME_COL].astype("int32")

    # inf -> NaN 收尾（衍生比率可能產生 inf）
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)

    cat_features = list(C.CATEGORICAL)
    feature_names = list(out.columns)
    return out, feature_names, cat_features
