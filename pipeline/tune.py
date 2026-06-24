# -*- coding: utf-8 -*-
"""Optuna 精調：每個模型以「OOF RMSE」為直接最佳化目標。

為控制時間，調參階段在**單一折、單一種子、較少樹數**上評估候選參數；
選出最佳參數後，再交給 models.py 用完整 5 折 × 3 種子產生正式 OOF。
"""
import warnings

import numpy as np
import optuna

from . import config as C
from .cv import rmse
from .models import _prep_catboost

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 調參階段的輕量設定（早停會自動截斷，故輪數只是上限）
TUNE_ROUNDS = 2500      # LightGBM / XGBoost 調參上限輪數
TUNE_ROUNDS_CAT = 1500  # CatBoost 較慢，調參輪數壓低
TUNE_ES = 120
TUNE_SEED = 42


# ---------------------------------------------------------------------------
# 各模型：單折訓練 + 驗證 RMSE
# ---------------------------------------------------------------------------
def _eval_lgbm(params, X, y, tr_idx, va_idx, cat_features):
    import lightgbm as lgb
    p = dict(objective="huber", metric="rmse", verbosity=-1, seed=TUNE_SEED, **params)
    dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_features)
    dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], categorical_feature=cat_features)
    model = lgb.train(p, dtrain, num_boost_round=TUNE_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(TUNE_ES, verbose=False), lgb.log_evaluation(0)])
    pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    return rmse(y[va_idx], pred)


def _eval_xgb(params, X, y, tr_idx, va_idx, cat_features):
    import xgboost as xgb
    p = dict(objective="reg:pseudohubererror", eval_metric="rmse",
             tree_method="hist", seed=TUNE_SEED, **params)
    dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
    dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
    model = xgb.train(p, dtrain, num_boost_round=TUNE_ROUNDS, evals=[(dvalid, "valid")],
                      early_stopping_rounds=TUNE_ES, verbose_eval=False)
    it = model.best_iteration
    pred = model.predict(dvalid, iteration_range=(0, it + 1))
    return rmse(y[va_idx], pred)


def _eval_cat(params, X, y, tr_idx, va_idx, cat_features):
    from catboost import CatBoostRegressor, Pool
    delta = params.pop("huber_delta")
    Xc = _prep_catboost(X, cat_features)
    train_pool = Pool(Xc.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
    valid_pool = Pool(Xc.iloc[va_idx], y[va_idx], cat_features=cat_features)
    model = CatBoostRegressor(
        iterations=TUNE_ROUNDS_CAT, random_seed=TUNE_SEED, early_stopping_rounds=TUNE_ES,
        allow_writing_files=False, verbose=False, thread_count=-1,
        loss_function=f"Huber:delta={delta}", eval_metric="RMSE", **params,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return rmse(y[va_idx], model.predict(valid_pool))


# ---------------------------------------------------------------------------
# 各模型的搜尋空間
# ---------------------------------------------------------------------------
def _suggest_lgbm(t):
    return dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.05, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 255),
        min_data_in_leaf=t.suggest_int("min_data_in_leaf", 50, 500, log=True),
        feature_fraction=t.suggest_float("feature_fraction", 0.5, 0.95),
        bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 0.95),
        bagging_freq=1,
        lambda_l1=t.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        lambda_l2=t.suggest_float("lambda_l2", 1e-3, 20.0, log=True),
        alpha=t.suggest_float("alpha", 10.0, 150.0, log=True),  # huber delta
    )


def _suggest_xgb(t):
    return dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.05, log=True),
        max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        subsample=t.suggest_float("subsample", 0.5, 0.95),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 0.95),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        huber_slope=t.suggest_float("huber_slope", 1.0, 100.0, log=True),
    )


def _suggest_cat(t):
    return dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.05, log=True),
        depth=t.suggest_int("depth", 4, 8),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        random_strength=t.suggest_float("random_strength", 0.0, 10.0),
        subsample=t.suggest_float("subsample", 0.5, 0.95),
        bootstrap_type="Bernoulli",
        huber_delta=t.suggest_float("huber_delta", 10.0, 150.0, log=True),
    )


_REGISTRY = {
    "lgbm": (_suggest_lgbm, _eval_lgbm),
    "xgb": (_suggest_xgb, _eval_xgb),
    "cat": (_suggest_cat, _eval_cat),
}


def tune_model(model_key, X, y, folds, cat_features, n_trials=None):
    """對單一模型做 Optuna 搜尋，回傳最佳參數 dict。

    使用 folds[0]（第一折）作為快速評估，single seed、較少樹數。
    """
    suggest, evaluate = _REGISTRY[model_key]
    tr_idx, va_idx = folds[0]
    n_trials = n_trials or C.OPTUNA_TRIALS

    def objective(trial):
        params = suggest(trial)
        return evaluate(dict(params), X, y, tr_idx, va_idx, cat_features)

    sampler = optuna.samplers.TPESampler(seed=C.RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=C.OPTUNA_TIMEOUT,
                   show_progress_bar=False)
    print(f"  [{model_key}] best single-fold RMSE = {study.best_value:.4f}")
    return study.best_params
