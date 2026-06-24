# 股票一年期報酬率預測 — OOF 診斷報告

- 訓練樣本數：23,070　特徵數：依 features.py 產生


## 1. Naive 基準線（模型須勝過）

- global_mean: RMSE = 138.6471
- sector_year_mean: RMSE = 133.9962

## 2. 各模型 OOF RMSE

- lgbm: RMSE = 128.6596　(集成權重 0.057)
- xgb: RMSE = 127.9450　(集成權重 0.626)
- cat: RMSE = 128.5481　(集成權重 0.108)
- mlp: RMSE = 129.4983　(集成權重 0.209)
- **集成 blend**: RMSE = 127.6991

## 3. 後處理逐步 OOF RMSE（最高槓桿）

- 集成後: 127.6991
- 偏誤校正後 (a=1.3097, b=-4.1320): 127.0141
- 向均值收縮後 (alpha=1.000): 127.0141
- 最佳裁剪後 (bounds=(-50.0, 1000.0)): 126.9966
- **最終 OOF RMSE = 126.9966**

## 4. RMSE 來源拆解（尾部 vs 主體）

- 集成後: 整體 127.70｜尾部top50 2487.42｜主體 53.89
- 最終: 整體 127.00｜尾部top50 2439.75｜主體 56.87

## 5. 輔助指標（最終預測）

- MAE = 37.6796
- Spearman（排序相關）= 0.5996

## 6. 分年 OOF RMSE（最終）

- 2019: 77.4514
- 2020: 239.5832
- 2021: 38.3757
- 2022: 63.2187

## 7. 分產業 OOF RMSE（最終）

- sector 0.0: 69.0513
- sector 1.0: 112.3609
- sector 2.0: 183.7402
- sector 3.0: 143.9558
- sector 4.0: 184.7617
- sector 5.0: 34.0681
- sector 6.0: 74.0800
- sector 7.0: 61.1235
- sector 8.0: 87.7103
- sector 9.0: 65.3442
- sector 10.0: 46.9270

## 8. Walk-forward 分年診斷（時間穩健度）

- 驗證 2020（集成後 blend）: RMSE = 263.5714
- 驗證 2021（集成後 blend）: RMSE = 64.0020
- 驗證 2022（集成後 blend）: RMSE = 64.1984

## 9. Optuna 最佳參數

- **lgbm**: `{'learning_rate': 0.04737050356526202, 'num_leaves': 59, 'min_data_in_leaf': 50, 'feature_fraction': 0.6267268882544025, 'bagging_fraction': 0.9456676107805093, 'lambda_l1': 2.1084660879383272, 'lambda_l2': 0.03640682737663628, 'alpha': 93.60637333255492}`
- **xgb**: `{'learning_rate': 0.02786246562533413, 'max_depth': 10, 'min_child_weight': 1.0176086225257894, 'subsample': 0.9365154642218252, 'colsample_bytree': 0.7268874700237842, 'reg_alpha': 0.1434897079365616, 'reg_lambda': 0.08927110551442854, 'huber_slope': 76.03008904779097}`
- **cat**: `{'learning_rate': 0.03539446918555809, 'depth': 7, 'l2_leaf_reg': 1.1924650806499018, 'random_strength': 5.195401279075572, 'subsample': 0.6857121862176606, 'huber_delta': 146.31722710091822}`
- **mlp**: `{'emb_dim': 4, 'hidden': (256, 128, 64), 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 512, 'max_epochs': 120, 'patience': 15, 'huber_beta': 1.0}`
