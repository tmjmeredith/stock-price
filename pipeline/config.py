# -*- coding: utf-8 -*-
"""全域設定：路徑、欄位清單、CV 與 Optuna 參數、隨機種子。

集中管理所有「不該散落在各處」的常數，讓 run.py 與各模組共用同一份設定。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑（以本檔位置為基準，往上一層即工作目錄 d:/kaggle/在試）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
SAMPLE_SUB_CSV = ROOT / "sample_submission.csv"
SUBMISSION_CSV = ROOT / "submission.csv"
OOF_REPORT_MD = ROOT / "oof_report.md"
ARTIFACT_DIR = ROOT / "artifacts"  # 暫存 OOF / 最佳參數等

# ---------------------------------------------------------------------------
# 欄位定義
# ---------------------------------------------------------------------------
TARGET = "return_pct"
ID_COL = "id"

# 不可作為特徵：身分欄、洩漏欄、目標
#   ticker        — 測試集匿名且與訓練集零重疊，身分無法泛化
#   period_start  — 測試集無此欄
#   period_end    — 測試集無此欄，且為「期末」屬未來資訊（洩漏）
DROP_COLS = ["id", "ticker", "period_start", "period_end", TARGET]

# 群組欄（CV 用，避免同一 ticker 跨折洩漏）
GROUP_COL = "ticker"
# 時間欄（walk-forward 診斷用）
TIME_COL = "start_year"

# 原始數值特徵（33 個基本面欄位，不含 id/ticker/年度/期間/目標/sector_code）
RAW_NUMERIC = [
    "pe_ttm", "price_to_book", "price_to_sales", "growth_pe_ratio",
    "gross_margin", "operating_margin", "net_margin", "roa", "roe", "rote",
    "revenue_growth_3y", "revenue_growth_yoy", "revenue_ttm", "net_income_ttm",
    "income_before_tax", "eps_basic", "eps_diluted", "total_assets",
    "stockholders_equity", "current_assets", "current_liabilities",
    "long_term_debt", "goodwill", "inventory", "current_ratio", "quick_ratio",
    "debt_to_equity", "dividend_yield", "dividends_ttm", "dividends_paid_ttm",
    "shares_outstanding", "shares_diluted",
]

# 類別特徵
CATEGORICAL = ["sector_code"]

# 橫斷面排名用的估值/品質比率（在每個 start_year 群組內取百分位排名）
RANK_FEATURES = [
    "pe_ttm", "price_to_book", "price_to_sales", "growth_pe_ratio",
    "gross_margin", "operating_margin", "net_margin", "roa", "roe", "rote",
    "revenue_growth_3y", "revenue_growth_yoy", "debt_to_equity",
    "current_ratio", "quick_ratio", "dividend_yield",
]

# 產業相對 z-score 用的關鍵比率（在 sector_code x start_year 內標準化）
SECTOR_REL_FEATURES = [
    "pe_ttm", "price_to_book", "price_to_sales", "roe", "roa",
    "net_margin", "operating_margin", "revenue_growth_yoy",
]

# 缺失指示閾值：缺失率高於此值才建 is_missing_ 欄
MISSING_FLAG_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# CV / 模型 / 調參設定
# ---------------------------------------------------------------------------
N_FOLDS = 5            # GroupKFold 折數
SEEDS = [42, 1337, 2025]  # seed ensembling 用的隨機種子
RANDOM_STATE = 42

# Optuna 設定（徹底精調）
# 各模型試驗數分開設定：LightGBM/XGBoost 快，給足 100+ 徹底搜尋；
# CatBoost 單次訓練慢（約為前兩者的 15 倍），給較精簡但仍充分的預算。
OPTUNA_TRIALS = 100    # 預設試驗數
OPTUNA_TRIALS_BY_MODEL = {"lgbm": 100, "xgb": 100, "cat": 35}
OPTUNA_TIMEOUT = None  # 不設秒數上限，以 trials 為準
ES_ROUNDS = 200        # 早停輪數
N_ESTIMATORS = 20000   # 上限樹數（搭配早停）

# 上一輪 Optuna 已搜出的最佳參數（--no-tune 時直接沿用，省去重新搜尋）
TUNED_PARAMS = {
    "lgbm": {"learning_rate": 0.04737050356526202, "num_leaves": 59,
             "min_data_in_leaf": 50, "feature_fraction": 0.6267268882544025,
             "bagging_fraction": 0.9456676107805093, "lambda_l1": 2.1084660879383272,
             "lambda_l2": 0.03640682737663628, "alpha": 93.60637333255492},
    "xgb": {"learning_rate": 0.02786246562533413, "max_depth": 10,
            "min_child_weight": 1.0176086225257894, "subsample": 0.9365154642218252,
            "colsample_bytree": 0.7268874700237842, "reg_alpha": 0.1434897079365616,
            "reg_lambda": 0.08927110551442854, "huber_slope": 76.03008904779097},
    "cat": {"learning_rate": 0.03539446918555809, "depth": 7,
            "l2_leaf_reg": 1.1924650806499018, "random_strength": 5.195401279075572,
            "subsample": 0.6857121862176606, "huber_delta": 146.31722710091822},
}
