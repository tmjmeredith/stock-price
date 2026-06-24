# -*- coding: utf-8 -*-
"""GBDT 模型包裝：LightGBM / XGBoost / CatBoost。

每個模型提供 `run_*` 函式：依給定 CV 折在原始尺度訓練（肥尾穩健損失），
做 seed ensembling，回傳全覆蓋 OOF 預測與測試集預測。

測試集預測採「折模型 bagging」：對每折 × 每種子訓練出的模型，在測試集上取平均。
此法穩健（不需在全資料上重挑早停輪數），且 OOF 與測試預測的生成過程一致。
"""
import warnings

import numpy as np
import pandas as pd

from . import config as C

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 預設參數（Optuna 未調參時的合理起點）
# ---------------------------------------------------------------------------
LGBM_DEFAULT = dict(
    objective="huber", alpha=30.0, metric="rmse",
    learning_rate=0.02, num_leaves=31, min_data_in_leaf=120,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=1.0, lambda_l2=5.0, max_depth=-1, verbosity=-1,
)

XGB_DEFAULT = dict(
    objective="reg:pseudohubererror", eval_metric="rmse", tree_method="hist",
    learning_rate=0.02, max_depth=6, min_child_weight=20.0,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
)

CAT_DEFAULT = dict(
    loss_function="Huber:delta=30", eval_metric="RMSE",
    learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
    random_strength=1.0, bootstrap_type="Bernoulli", subsample=0.8,
    thread_count=-1,
)


# ---------------------------------------------------------------------------
# 各模型的類別欄前處理
# ---------------------------------------------------------------------------
def _prep_catboost(X, cat_features):
    """CatBoost：類別欄需為字串且不含 NaN。"""
    X = X.copy()
    for c in cat_features:
        # 字串類別 -> 數值 -> 補 -1 -> 字串；CatBoost 類別欄需為字串且不含 NaN。
        num = pd.to_numeric(X[c].astype("object"), errors="coerce")
        X[c] = num.fillna(-1).astype("int").astype("str")
    return X


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def run_lgbm(X, y, X_test, folds, params=None, cat_features=None, seeds=None):
    import lightgbm as lgb

    params = {**LGBM_DEFAULT, **(params or {})}
    seeds = seeds or C.SEEDS
    cat_features = cat_features or []

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_fold = len(folds)
    n_seed = len(seeds)
    best_iters = []

    for tr_idx, va_idx in folds:
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        dvalid = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features)
        fold_va = np.zeros(len(va_idx))
        for seed in seeds:
            p = {**params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
            model = lgb.train(
                p, dtrain, num_boost_round=C.N_ESTIMATORS,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(C.ES_ROUNDS, verbose=False),
                           lgb.log_evaluation(0)],
            )
            best_iters.append(model.best_iteration)
            fold_va += model.predict(X_va, num_iteration=model.best_iteration) / n_seed
            test_pred += model.predict(X_test, num_iteration=model.best_iteration) / (n_fold * n_seed)
        oof[va_idx] = fold_va
    return oof, test_pred, {"best_iters": best_iters}


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def run_xgb(X, y, X_test, folds, params=None, cat_features=None, seeds=None):
    import xgboost as xgb

    params = {**XGB_DEFAULT, **(params or {})}
    seeds = seeds or C.SEEDS

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_fold = len(folds)
    n_seed = len(seeds)
    best_iters = []

    # enable_categorical 需要 category 型別；建立一次 DMatrix 重用
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    for tr_idx, va_idx in folds:
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        fold_va = np.zeros(len(va_idx))
        for seed in seeds:
            p = {**params, "seed": seed}
            model = xgb.train(
                p, dtrain, num_boost_round=C.N_ESTIMATORS,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=C.ES_ROUNDS, verbose_eval=False,
            )
            best_iters.append(model.best_iteration)
            it = model.best_iteration
            fold_va += model.predict(dvalid, iteration_range=(0, it + 1)) / n_seed
            test_pred += model.predict(dtest, iteration_range=(0, it + 1)) / (n_fold * n_seed)
        oof[va_idx] = fold_va
    return oof, test_pred, {"best_iters": best_iters}


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------
def run_cat(X, y, X_test, folds, params=None, cat_features=None, seeds=None):
    from catboost import CatBoostRegressor, Pool

    params = {**CAT_DEFAULT, **(params or {})}
    seeds = seeds or C.SEEDS
    cat_features = cat_features or []

    Xc = _prep_catboost(X, cat_features)
    Xtest_c = _prep_catboost(X_test, cat_features)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_fold = len(folds)
    n_seed = len(seeds)
    best_iters = []

    test_pool = Pool(Xtest_c, cat_features=cat_features)

    for tr_idx, va_idx in folds:
        X_tr, y_tr = Xc.iloc[tr_idx], y[tr_idx]
        X_va, y_va = Xc.iloc[va_idx], y[va_idx]
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        valid_pool = Pool(X_va, y_va, cat_features=cat_features)
        fold_va = np.zeros(len(va_idx))
        for seed in seeds:
            model = CatBoostRegressor(
                iterations=C.N_ESTIMATORS, random_seed=seed,
                early_stopping_rounds=C.ES_ROUNDS, allow_writing_files=False,
                verbose=False, **params,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            best_iters.append(model.get_best_iteration())
            fold_va += model.predict(valid_pool) / n_seed
            test_pred += model.predict(test_pool) / (n_fold * n_seed)
        oof[va_idx] = fold_va
    return oof, test_pred, {"best_iters": best_iters}


# 模型登記表，供 run.py / tune.py 統一呼叫
MODEL_RUNNERS = {
    "lgbm": run_lgbm,
    "xgb": run_xgb,
    "cat": run_cat,
}
