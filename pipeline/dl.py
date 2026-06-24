# -*- coding: utf-8 -*-
"""深度學習模型：強正則化 MLP + sector embedding。

提供與 GBDT 相同介面的 run_mlp，產生全覆蓋 OOF 與測試集預測，可直接併入集成。

DL 前處理要點（與 GBDT 不同，DL 對尺度/缺失敏感）：
- 連續數值：每折在訓練列上 fit 中位數補值 + QuantileTransformer(常態)，避免極端值發散。
- 缺失指示（is_missing_*）：保留 0/1，不做分位轉換。
- sector_code：以 embedding 學習（NaN 自成一類）。
- 目標 return_pct：每折用訓練列的 mean/std 標準化後，以 Huber(SmoothL1) 損失訓練，
  預測後反標準化回原尺度。標準化 + Huber 兼顧優化穩定與抗肥尾。
"""
import numpy as np
import pandas as pd

from . import config as C
from .cv import rmse


# ---------------------------------------------------------------------------
# 預設超參數（固定合理值 + seed ensembling；DL 對單組好參數已相當穩健）
# ---------------------------------------------------------------------------
MLP_DEFAULT = dict(
    emb_dim=4, hidden=(256, 128, 64), dropout=0.3,
    lr=1e-3, weight_decay=1e-5, batch_size=512, max_epochs=120,
    patience=15, huber_beta=1.0,
)


def _split_columns(X, cat_features):
    """切出 sector 類別欄、連續數值欄、缺失旗標欄。"""
    flag_cols = [c for c in X.columns if c.startswith("is_missing_")]
    cont_cols = [c for c in X.columns
                 if c not in cat_features and c not in flag_cols]
    return cont_cols, flag_cols


def _sector_codes(series):
    """字串類別 -> 整數索引；NaN -> 0，其餘 1..K。"""
    codes = series.cat.codes.values.astype("int64")  # NaN 為 -1
    return codes + 1  # 平移到 0..K，0 代表缺失/未知


def run_mlp(X, y, X_test, folds, params=None, cat_features=None, seeds=None):
    import torch
    import torch.nn as nn
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer

    params = {**MLP_DEFAULT, **(params or {})}
    seeds = seeds or C.SEEDS
    cat_features = cat_features or []
    sector_col = cat_features[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cont_cols, flag_cols = _split_columns(X, cat_features)

    # sector embedding 基數（train/test 已統一類別）
    n_sector = int(max(_sector_codes(X[sector_col]).max(),
                       _sector_codes(X_test[sector_col]).max())) + 1

    # 預先取出旗標與 sector（不需逐折 fit）
    flags_all = X[flag_cols].fillna(0).values.astype("float32")
    flags_test = X_test[flag_cols].fillna(0).values.astype("float32")
    sec_all = _sector_codes(X[sector_col])
    sec_test = _sector_codes(X_test[sector_col])

    class MLP(nn.Module):
        def __init__(self, n_cont, n_flag, n_sec, emb_dim, hidden, dropout):
            super().__init__()
            self.emb = nn.Embedding(n_sec, emb_dim)
            in_dim = n_cont + n_flag + emb_dim
            layers = []
            for h in hidden:
                layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)]
                in_dim = h
            layers += [nn.Linear(in_dim, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x_cont, x_flag, x_sec):
            e = self.emb(x_sec)
            x = torch.cat([x_cont, x_flag, e], dim=1)
            return self.net(x).squeeze(1)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    n_fold = len(folds)
    n_seed = len(seeds)

    for tr_idx, va_idx in folds:
        # 連續特徵：折內 fit 補值 + 分位轉換
        imp = SimpleImputer(strategy="median")
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=min(1000, len(tr_idx)),
                                 subsample=10**9, random_state=C.RANDOM_STATE)
        cont_tr = qt.fit_transform(imp.fit_transform(X.iloc[tr_idx][cont_cols]))
        cont_va = qt.transform(imp.transform(X.iloc[va_idx][cont_cols]))
        cont_te = qt.transform(imp.transform(X_test[cont_cols]))

        # 目標標準化
        y_tr = y[tr_idx]
        y_mean, y_std = float(y_tr.mean()), float(y_tr.std() + 1e-8)
        z_tr = (y_tr - y_mean) / y_std

        # 轉 tensor
        def to_t(a):
            return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)

        Xc_tr, Xf_tr = to_t(cont_tr), to_t(flags_all[tr_idx])
        Xs_tr = torch.tensor(sec_all[tr_idx], dtype=torch.long, device=device)
        zt = to_t(z_tr)
        Xc_va, Xf_va = to_t(cont_va), to_t(flags_all[va_idx])
        Xs_va = torch.tensor(sec_all[va_idx], dtype=torch.long, device=device)
        Xc_te, Xf_te = to_t(cont_te), to_t(flags_test)
        Xs_te = torch.tensor(sec_test, dtype=torch.long, device=device)

        fold_va = np.zeros(len(va_idx))
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MLP(len(cont_cols), len(flag_cols), n_sector,
                        params["emb_dim"], params["hidden"], params["dropout"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                   weight_decay=params["weight_decay"])
            lossf = nn.SmoothL1Loss(beta=params["huber_beta"])
            n = len(tr_idx)
            bs = params["batch_size"]
            best_rmse, best_state, bad = np.inf, None, 0
            for epoch in range(params["max_epochs"]):
                model.train()
                perm = torch.randperm(n, device=device)
                for i in range(0, n, bs):
                    b = perm[i:i + bs]
                    opt.zero_grad()
                    out = model(Xc_tr[b], Xf_tr[b], Xs_tr[b])
                    loss = lossf(out, zt[b])
                    loss.backward()
                    opt.step()
                # 驗證（原尺度 RMSE）
                model.eval()
                with torch.no_grad():
                    pv = model(Xc_va, Xf_va, Xs_va).cpu().numpy() * y_std + y_mean
                r = rmse(y[va_idx], pv)
                if r < best_rmse - 1e-4:
                    best_rmse, best_state, bad = r, {k: v.detach().clone()
                                                     for k, v in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= params["patience"]:
                        break
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                pv = model(Xc_va, Xf_va, Xs_va).cpu().numpy() * y_std + y_mean
                pt = model(Xc_te, Xf_te, Xs_te).cpu().numpy() * y_std + y_mean
            fold_va += pv / n_seed
            test_pred += pt / (n_fold * n_seed)
        oof[va_idx] = fold_va
    return oof, test_pred, {}
