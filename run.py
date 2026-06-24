# -*- coding: utf-8 -*-
"""主程式：串接全流程，輸出 submission.csv 與 oof_report.md。

執行（imaginary 環境）：
    C:/Users/tfx92/anaconda3/envs/imaginary/python.exe run.py
快速煙霧測試（少量 trials / seeds，驗證流程不出錯）：
    C:/Users/tfx92/anaconda3/envs/imaginary/python.exe run.py --smoke
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd

from pipeline import config as C
from pipeline import data as D
from pipeline import cv
from pipeline import features as F
from pipeline import models as M
from pipeline import tune as T
from pipeline import dl as DL
from pipeline import postprocess as PP
from pipeline import report as R

warnings.filterwarnings("ignore")

SMOKE = "--smoke" in sys.argv
NO_TUNE = "--no-tune" in sys.argv   # 沿用 config.TUNED_PARAMS，跳過 Optuna
USE_DL = "--dl" in sys.argv         # 加入 MLP 進集成（需明確指定 --dl）
if SMOKE:
    # 煙霧測試：壓低成本，只為驗證流程串接正確
    C.OPTUNA_TRIALS = 5
    C.OPTUNA_TRIALS_BY_MODEL = {"lgbm": 5, "xgb": 5, "cat": 5}
    C.SEEDS = [42]
    C.N_FOLDS = 3
    C.N_ESTIMATORS = 2000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unify_categoricals(X_tr, X_te, cat_features):
    """統一 train/test 的類別欄編碼，避免 XGB/LGBM 類別錯位。"""
    for c in cat_features:
        cats = pd.api.types.union_categoricals(
            [X_tr[c].astype("category"), X_te[c].astype("category")]
        ).categories
        X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
        X_te[c] = pd.Categorical(X_te[c], categories=cats)
    return X_tr, X_te


def compute_baselines(train_df, test_df):
    """計算向均值收縮用的 baseline。

    測試集年度（2024）無標籤，故 baseline 採「產業層級均值」（測試集可重現）。
    - baseline_oof：對每列用「其產業在全訓練集的均值」（平滑、低過擬合風險）。
    - baseline_test：用全訓練集的產業均值，依測試集 sector_code 對映。
    """
    glob = train_df[C.TARGET].mean()
    sector_mean = train_df.groupby(C.CATEGORICAL[0])[C.TARGET].mean()
    base_oof = train_df[C.CATEGORICAL[0]].map(sector_mean).fillna(glob).values
    base_test = test_df[C.CATEGORICAL[0]].map(sector_mean).fillna(glob).values
    return base_oof, base_test


def main():
    t0 = time.time()
    log(f"開始（SMOKE={SMOKE}）")

    # ---- 載入與清理 ----
    train_raw, test_raw, sample_sub = D.load_raw()
    train_raw = D.basic_clean(train_raw)
    test_raw = D.basic_clean(test_raw)
    y = train_raw[C.TARGET].values.astype(float)
    log(f"train={train_raw.shape}  test={test_raw.shape}")

    # ---- 特徵工程（train/test 共用同一函式） ----
    X, feat_names, cat_features = F.build_features(train_raw)
    X_test, _, _ = F.build_features(test_raw)
    X, X_test = unify_categoricals(X, X_test, cat_features)
    log(f"特徵數={X.shape[1]}  類別特徵={cat_features}")

    # ---- CV 折與基準線 ----
    folds = cv.group_folds(train_raw)
    baselines = cv.naive_baselines(train_raw)
    for k, v in baselines.items():
        log(f"  baseline {k}: RMSE={v:.4f}")
    base_oof, base_test = compute_baselines(train_raw, test_raw)

    # ---- 各模型：Optuna 精調 → 完整 CV 產生 OOF + test ----
    per_model_oof = {}
    per_model_test = {}
    model_params = {}
    for mk in ["lgbm", "xgb", "cat"]:
        if NO_TUNE:
            best = dict(C.TUNED_PARAMS[mk])
            log(f"=== 模型 {mk}：沿用 TUNED_PARAMS（跳過 Optuna）===")
        else:
            n_trials = C.OPTUNA_TRIALS_BY_MODEL.get(mk, C.OPTUNA_TRIALS)
            log(f"=== 模型 {mk}：Optuna 精調（{n_trials} trials）===")
            best = T.tune_model(mk, X, y, folds, cat_features, n_trials=n_trials)
        # CatBoost 參數轉換：huber_delta -> loss_function 字串
        run_params = dict(best)
        if mk == "cat" and "huber_delta" in run_params:
            delta = run_params.pop("huber_delta")
            run_params["loss_function"] = f"Huber:delta={delta}"
        model_params[mk] = best
        log(f"=== 模型 {mk}：完整 {C.N_FOLDS} 折 × {len(C.SEEDS)} 種子訓練 ===")
        oof, test_pred, info = M.MODEL_RUNNERS[mk](
            X, y, X_test, folds, params=run_params,
            cat_features=cat_features, seeds=C.SEEDS,
        )
        per_model_oof[mk] = oof
        per_model_test[mk] = test_pred
        log(f"  {mk} OOF RMSE = {cv.rmse(y, oof):.4f}")

    # ---- 深度學習 MLP（加入集成增加誤差多樣性） ----
    if USE_DL:
        log(f"=== 模型 mlp：MLP + sector embedding（{C.N_FOLDS} 折 × {len(C.SEEDS)} 種子）===")
        oof, test_pred, _ = DL.run_mlp(X, y, X_test, folds,
                                       cat_features=cat_features, seeds=C.SEEDS)
        per_model_oof["mlp"] = oof
        per_model_test["mlp"] = test_pred
        model_params["mlp"] = dict(DL.MLP_DEFAULT)
        log(f"  mlp OOF RMSE = {cv.rmse(y, oof):.4f}")

    # ---- 保存各模型 OOF/test 預測（供日後免重訓重新集成） ----
    try:
        C.ARTIFACT_DIR.mkdir(exist_ok=True)
        np.savez(C.ARTIFACT_DIR / "preds.npz",
                 y=y, test_id=test_raw[C.ID_COL].values,
                 **{f"oof_{k}": v for k, v in per_model_oof.items()},
                 **{f"test_{k}": v for k, v in per_model_test.items()})
        log(f"已保存模型預測到 {C.ARTIFACT_DIR / 'preds.npz'}")
    except Exception as e:
        log(f"保存 artifacts 失敗（不影響主流程）：{e}")

    # ---- 集成（OOF 上求非負權重） ----
    weights, blend_oof = PP.optimize_ensemble_weights(per_model_oof, y)
    blend_test = PP.apply_weights(per_model_test, weights)
    log(f"集成權重={ {k: round(v,3) for k,v in weights.items()} }  blend OOF RMSE={cv.rmse(y, blend_oof):.4f}")

    # ---- 後處理：偏誤校正 → 收縮 → 裁剪（全由 OOF 鎖定） ----
    post = PP.PostProcessor()
    final_oof = post.fit(blend_oof, y, base_oof)
    final_test = post.transform(blend_test, base_test)
    log(f"後處理鏈 OOF：{ {k: (round(v,4) if isinstance(v,(int,float)) else v) for k,v in post.history.items()} }")
    log(f"最終 OOF RMSE = {cv.rmse(y, final_oof):.4f}")

    # ---- Walk-forward 時間穩健診斷（以 LightGBM 代表模型，單種子） ----
    wf_results = {}
    for year, tr_idx, va_idx in cv.walk_forward_folds(train_raw):
        r = T._eval_lgbm(dict(model_params["lgbm"]), X, y, tr_idx, va_idx, cat_features)
        wf_results[year] = r
        log(f"  walk-forward 驗證 {year}: RMSE={r:.4f}")

    # ---- 產出提交檔（對齊 sample_submission 的 id 順序） ----
    final_test = np.nan_to_num(final_test, nan=float(y.mean()),
                               posinf=float(np.nanmax(y)), neginf=float(np.nanmin(y)))
    sub = pd.DataFrame({C.ID_COL: test_raw[C.ID_COL].values, C.TARGET: final_test})
    sub = sample_sub[[C.ID_COL]].merge(sub, on=C.ID_COL, how="left")
    assert sub[C.TARGET].isna().sum() == 0, "提交檔有 NaN"
    assert len(sub) == len(sample_sub), "提交檔列數不符"
    sub.to_csv(C.SUBMISSION_CSV, index=False)
    log(f"已寫出 {C.SUBMISSION_CSV}（{len(sub)} 列）")

    # ---- 診斷報告 ----
    R.build_report(train_raw, y, per_model_oof, blend_oof, final_oof, baselines,
                   weights, post.history, wf_results, model_params)
    log(f"已寫出 {C.OOF_REPORT_MD}")
    log(f"完成，總耗時 {(time.time()-t0)/60:.1f} 分鐘")


if __name__ == "__main__":
    main()
