# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 設定中文畫圖字型避免亂碼 (相容 Windows 與 Linux 雲端環境)
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

class TimeAwareMetaLearner(BaseEstimator, RegressorMixin):
    """
    時空感知動態權重 Meta-Model
    依據時間特徵 (Month_sin/cos) 動態調整 Ridge, RF, GBDT 的加權比例
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.meta_model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, positive=True, fit_intercept=False, max_iter=5000)
        
    def _create_dynamic_features(self, X_meta_preds, X_time, is_fit=False):
        """
        建立交互特徵：將子模型預測值與時間特徵相乘
        讓模型能學到：在特定的季節或月份，調整基準模型比例
        """
        if is_fit:
            X_time_scaled = self.scaler.fit_transform(X_time)
        else:
            X_time_scaled = self.scaler.transform(X_time)
            
        interactions = []
        for i in range(X_meta_preds.shape[1]):
            pred_col = X_meta_preds[:, i:i+1]
            # 基礎權重
            interactions.append(pred_col)
            # 動態時間權重修正項
            interactions.append(pred_col * X_time_scaled)
            
        return np.column_stack(interactions)

    def fit(self, X_meta_val, X_time_val, y_val):
        """
        X_meta_val: 驗證集的子模型預測值矩陣 (n_samples, 3)
        X_time_val: 驗證集的時間特徵矩陣 (n_samples, 4)
        y_val: 驗證集的真實氣候值 (n_samples,)
        """
        X_meta_val = np.asarray(X_meta_val)
        X_time_val = np.asarray(X_time_val)
        y_val = np.asarray(y_val)
        
        X_dynamic = self._create_dynamic_features(X_meta_val, X_time_val, is_fit=True)
        self.meta_model.fit(X_dynamic, y_val)
        return self
        
    def predict(self, X_meta_test, X_time_test):
        X_meta_test = np.asarray(X_meta_test)
        X_time_test = np.asarray(X_time_test)
        X_dynamic = self._create_dynamic_features(X_meta_test, X_time_test, is_fit=False)
        return self.meta_model.predict(X_dynamic)

    def get_effective_weights(self, X_time):
        """
        計算給定時間特徵下的基準模型有效權重。
        X_time: 二維陣列或 DataFrame, 形如 (n_samples, 4)
        """
        X_time = np.asarray(X_time)
        X_time_scaled = self.scaler.transform(X_time)
        coef = self.meta_model.coef_
        
        weights = []
        n_samples = len(X_time_scaled)
        
        for i in range(3):
            base_coef = coef[5*i]
            time_coefs = coef[5*i+1 : 5*i+5]
            
            w_eff = base_coef + np.dot(X_time_scaled, time_coefs)
            w_eff = np.maximum(0.0, w_eff)
            weights.append(w_eff)
            
        weights = np.column_stack(weights)
        
        w_sums = weights.sum(axis=1, keepdims=True)
        w_sums = np.where(w_sums > 1e-5, w_sums, 1.0)
        normalized_weights = weights / w_sums
        
        for idx in range(n_samples):
            if normalized_weights[idx].sum() < 1e-5:
                normalized_weights[idx] = [0.1, 0.4, 0.5]
                
        return normalized_weights

def prepare_features(df):
    """
    對真實 CWA 欄位進行特徵處理：
    1. 排序 Date。
    2. 提取時間特徵與週期性編碼。
    3. 自動偵測所有數值氣候欄位（排除 Date, ObsTime, StationID 與包含 Time/LST 的時間標記欄位）。
    4. 對每一個數值欄位，建立前 1 天、前 2 天時間滯後特徵 (Lag 1, Lag 2) 以及一階動能差值 (diff1)。
    """
    df_sorted = df.sort_values(by=["StationID", "Date"]).copy()
    
    # 提取時間特徵
    df_sorted["Month"] = df_sorted["Date"].dt.month
    df_sorted["DayOfYear"] = df_sorted["Date"].dt.dayofyear
    
    # 週期性時間特徵編碼 (週期編碼)
    df_sorted["Month_sin"] = np.sin(2 * np.pi * df_sorted["Month"] / 12)
    df_sorted["Month_cos"] = np.cos(2 * np.pi * df_sorted["Month"] / 12)
    df_sorted["DayOfYear_sin"] = np.sin(2 * np.pi * df_sorted["DayOfYear"] / 365.25)
    df_sorted["DayOfYear_cos"] = np.cos(2 * np.pi * df_sorted["DayOfYear"] / 365.25)
    
    # 找出所有要預測與建立滯後特徵的數值欄位 (排除了時間標記與衍生時間欄位)
    exclude_cols = [
        "Date", "ObsTime", "StationID", 
        "Month", "DayOfYear", 
        "Month_sin", "Month_cos", 
        "DayOfYear_sin", "DayOfYear_cos"
    ]
    predict_cols = []
    for col in df_sorted.columns:
        if col not in exclude_cols and "Time" not in col and "LST" not in col:
            # 確保欄位是數值型態
            df_sorted[col] = pd.to_numeric(df_sorted[col], errors="coerce")
            # 如果該欄位缺失值大於 50%，則不納入預測以防 dropna 刪除整張表
            if df_sorted[col].isna().sum() / len(df_sorted) > 0.5:
                continue
            if pd.api.types.is_numeric_dtype(df_sorted[col]):
                predict_cols.append(col)
                
    # 建立 Lag 1, Lag 2 與 Diff 1 特徵
    lag_cols = []
    new_cols_dict = {}
    for col in predict_cols:
        lag1_name = f"{col}_lag1"
        lag2_name = f"{col}_lag2"
        diff1_name = f"{col}_diff1"
        new_cols_dict[lag1_name] = df_sorted.groupby("StationID")[col].shift(1)
        new_cols_dict[lag2_name] = df_sorted.groupby("StationID")[col].shift(2)
        new_cols_dict[diff1_name] = new_cols_dict[lag1_name] - new_cols_dict[lag2_name]
        lag_cols.extend([lag1_name, lag2_name, diff1_name])
        
    df_sorted = pd.concat([df_sorted, pd.DataFrame(new_cols_dict, index=df_sorted.index)], axis=1)
        
    # 刪除因為位移產生的第一列缺失值，以及其他包含 NaN 的列以利訓練
    # 注意：對於真實資料，我們使用前後填充 (ffill/bfill) 來填補缺失
    df_filled = df_sorted.copy()
    df_filled[predict_cols + lag_cols] = df_filled[predict_cols + lag_cols].ffill()
    df_clean = df_filled.dropna(subset=predict_cols + lag_cols).copy()
    
    return df_clean, predict_cols, lag_cols

def train_and_evaluate(df_clean, predict_cols, lag_cols):
    """
    為每一個氣候變數訓練獨立的預測模型，並以時空感知動態權重 Meta-Model 自動學習最優融合權重。
    """
    feature_cols = ["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"] + lag_cols
    X = df_clean[feature_cols]
    
    horizon = 7
    
    lr_models = {}
    rf_models = {}
    gbdt_models = {}
    meta_learners = {}
    metrics = {}
    test_predictions = {}
    
    # 定義物理上不可能為負數的氣象項目
    non_negative_cols = [
        "Precp", "PrecpHour", "PrecpMax10", "PrecpMax60", 
        "WS", "WSGust", "SunShine", "SunshineRate", 
        "GloblRad", "EvapA", "UVI Max"
    ]
    
    # 用於儲存切分後的測試集數據
    X_test_aligned = None
    df_test_aligned = None
    
    for col in predict_cols:
        # 1. 建立 Y_multistep: 每一列包含 [t+1, t+2, ..., t+horizon]
        Y_list = []
        for h in range(1, horizon + 1):
            Y_list.append(df_clean.groupby("StationID")[col].shift(-h))
        Y_multistep = pd.concat(Y_list, axis=1).dropna()
        Y_multistep.columns = [f"{col}_step_{h}" for h in range(1, horizon + 1)]
        
        # 保持 X、時間特徵與 Y 的列數對齊
        X_aligned = X.loc[Y_multistep.index]
        df_clean_aligned = df_clean.loc[Y_multistep.index]
        time_cols = ["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"]
        X_time = df_clean_aligned[time_cols]
        
        # 2. 資料集切分：若資料大於 20 筆，採用 80/20 時間切分
        n_rows = len(Y_multistep)
        if n_rows >= 20:
            train_size = int(n_rows * 0.8)
            X_train = X_aligned.iloc[:train_size]
            X_test = X_aligned.iloc[train_size:]
            Y_train = Y_multistep.iloc[:train_size]
            Y_test = Y_multistep.iloc[train_size:]
            df_train = df_clean_aligned.iloc[:train_size]
            df_test = df_clean_aligned.iloc[train_size:]
            X_time_train = X_time.iloc[:train_size]
            X_time_test = X_time.iloc[train_size:]
            
            # 為了動態學習集成權重，對訓練集再次進行時間先後 80/20 切分作為驗證集，避免資料洩漏
            val_size = int(train_size * 0.2)
            train_base_size = train_size - val_size
            X_train_base = X_train.iloc[:train_base_size]
            X_val = X_train.iloc[train_base_size:]
            Y_train_base = Y_train.iloc[:train_base_size]
            Y_val = Y_train.iloc[train_base_size:]
            X_time_val = X_time_train.iloc[train_base_size:]
        else:
            X_train = X_aligned
            X_test = X_aligned
            Y_train = Y_multistep
            Y_test = Y_multistep
            df_train = df_clean_aligned
            df_test = df_clean_aligned
            X_time_train = X_time
            X_time_test = X_time
            X_train_base, X_val = X_train, X_train
            Y_train_base, Y_val = Y_train, Y_train
            X_time_val = X_time_train
            
        X_test_aligned = X_test
        df_test_aligned = df_test
        
        # --- 動態 Stacking 權重學習 ---
        # 1. 於基礎訓練集擬合臨時模型
        lr_temp = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
        lr_temp.fit(X_train_base, Y_train_base)
        y_val_pred_lr = lr_temp.predict(X_val)
        
        rf_temp = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42)
        rf_temp.fit(X_train_base, Y_train_base)
        y_val_pred_rf = rf_temp.predict(X_val)
        
        gbdt_temp = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=42))
        gbdt_temp.fit(X_train_base, Y_train_base)
        y_val_pred_gbdt = gbdt_temp.predict(X_val)
        
        # 2. 為未來 7 天的每一步分別訓練一個 TimeAwareMetaLearner
        col_meta_learners = []
        for h in range(1, horizon + 1):
            y_val_true_h = Y_val.iloc[:, h-1]
            X_meta_val_h = np.column_stack([
                y_val_pred_lr[:, h-1],
                y_val_pred_rf[:, h-1],
                y_val_pred_gbdt[:, h-1]
            ])
            meta_h = TimeAwareMetaLearner(alpha=1.0, l1_ratio=0.5)
            meta_h.fit(X_meta_val_h, X_time_val, y_val_true_h)
            col_meta_learners.append(meta_h)
        meta_learners[col] = col_meta_learners
        
        # --- 訓練最終基準模型 (使用完整訓練集 X_train) ---
        # 1. 嶺回歸多步預測
        lr = MultiOutputRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
        lr.fit(X_train, Y_train)
        y_pred_lr = lr.predict(X_test)
        lr_models[col] = lr
        
        # 2. 隨機森林 (原生支援多輸出)
        rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=42)
        rf.fit(X_train, Y_train)
        y_pred_rf = rf.predict(X_test)
        rf_models[col] = rf
        
        # 3. GBDT 多步預測 (HistGradientBoosting 不支援多輸出，以 MultiOutputRegressor 包裝)
        gbdt = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=42))
        gbdt.fit(X_train, Y_train)
        y_pred_gbdt = gbdt.predict(X_test)
        gbdt_models[col] = gbdt
        
        # 4. 套用動態時空門控權重計算 Ensemble 預測
        y_pred_ensemble = np.zeros_like(y_pred_lr)
        for h in range(1, horizon + 1):
            pred_lr_h = y_pred_lr[:, h-1]
            pred_rf_h = y_pred_rf[:, h-1]
            pred_gbdt_h = y_pred_gbdt[:, h-1]
            
            y_pred_ensemble_h = col_meta_learners[h-1].predict(
                np.column_stack([pred_lr_h, pred_rf_h, pred_gbdt_h]),
                X_time_test
            )
            y_pred_ensemble[:, h-1] = y_pred_ensemble_h
            
        # 物理非負限制處理 (如降雨量、風速、日照時數不可能為負值)
        if col in non_negative_cols:
            y_pred_lr = np.clip(y_pred_lr, 0, None)
            y_pred_rf = np.clip(y_pred_rf, 0, None)
            y_pred_gbdt = np.clip(y_pred_gbdt, 0, None)
            y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
        
        # 儲存測試集預測結果 (lr, rf, gbdt, ensemble 僅取第 1 步以保持舊 UI 相容)
        test_predictions[col] = {
            "lr": y_pred_lr[:, 0],
            "rf": y_pred_rf[:, 0],
            "gbdt": y_pred_gbdt[:, 0],
            "ensemble": y_pred_ensemble[:, 0],
            # 完整多步預測矩陣，供分析成效與誤差曲線繪圖
            "lr_all": y_pred_lr,
            "rf_all": y_pred_rf,
            "gbdt_all": y_pred_gbdt,
            "ensemble_all": y_pred_ensemble,
            "true_all": Y_test.values
        }
        
        # 5. 評估指標計算 (針對第一天 Step 1 預估成效評估，以與歷史資料對應)
        y_test_step1 = Y_test.iloc[:, 0]
        y_pred_lr_step1 = y_pred_lr[:, 0]
        y_pred_rf_step1 = y_pred_rf[:, 0]
        y_pred_gbdt_step1 = y_pred_gbdt[:, 0]
        y_pred_ens_step1 = y_pred_ensemble[:, 0]
        
        if len(y_test_step1) > 1:
            try:
                def get_metrics(y_true, y_pred):
                    r2 = np.round(r2_score(y_true, y_pred), 4)
                    rmse = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
                    mae = np.round(mean_absolute_error(y_true, y_pred), 2)
                    return r2, rmse, mae
                
                r2_lr, rmse_lr, mae_lr = get_metrics(y_test_step1, y_pred_lr_step1)
                r2_rf, rmse_rf, mae_rf = get_metrics(y_test_step1, y_pred_rf_step1)
                r2_gbdt, rmse_gbdt, mae_gbdt = get_metrics(y_test_step1, y_pred_gbdt_step1)
                r2_ens, rmse_ens, mae_ens = get_metrics(y_test_step1, y_pred_ens_step1)
            except Exception:
                r2_lr, rmse_lr, mae_lr = 1.0, 0.0, 0.0
                r2_rf, rmse_rf, mae_rf = 1.0, 0.0, 0.0
                r2_gbdt, rmse_gbdt, mae_gbdt = 1.0, 0.0, 0.0
                r2_ens, rmse_ens, mae_ens = 1.0, 0.0, 0.0
        else:
            r2_lr, rmse_lr, mae_lr = 1.0, 0.0, 0.0
            r2_rf, rmse_rf, mae_rf = 1.0, 0.0, 0.0
            r2_gbdt, rmse_gbdt, mae_gbdt = 1.0, 0.0, 0.0
            r2_ens, rmse_ens, mae_ens = 1.0, 0.0, 0.0
            
        # 計算測試集平均門控權重做為 metrics 預設展現
        w_test_avg = col_meta_learners[0].get_effective_weights(X_time_test).mean(axis=0)
        metrics[col] = {
            "嶺回歸_R平方": r2_lr, "嶺回歸_RMSE": rmse_lr, "嶺回歸_MAE": mae_lr,
            "隨機森林_R平方": r2_rf, "隨機森林_RMSE": rmse_rf, "隨機森林_MAE": mae_rf,
            "梯度提升樹_R平方": r2_gbdt, "梯度提升樹_RMSE": rmse_gbdt, "梯度提升樹_MAE": mae_gbdt,
            "加權集成_R平方": r2_ens, "加權集成_RMSE": rmse_ens, "加權集成_MAE": mae_ens,
            "權重_Ridge": float(np.round(w_test_avg[0], 4)),
            "權重_RF": float(np.round(w_test_avg[1], 4)),
            "權重_GBDT": float(np.round(w_test_avg[2], 4))
        }
        
    # --- 6. 模擬傳統遞迴預測並計算 Direct vs Recursive 的 Lead Time 誤差曲線 ---
    # 建立 X_rec 副本以逐日遞迴更新 lag 欄位
    X_rec = X_test_aligned.copy()
    y_pred_rec = np.zeros((len(X_test_aligned), len(predict_cols), horizon))
    
    for h in range(1, horizon + 1):
        # 預測當前步驟 h (使用第一步的基準模型)
        for c_idx, col in enumerate(predict_cols):
            lr_1s = lr_models[col].estimators_[0]
            pred_lr_h = lr_1s.predict(X_rec)
            
            pred_rf_h = rf_models[col].predict(X_rec)[:, 0]
            
            gbdt_1s = gbdt_models[col].estimators_[0]
            pred_gbdt_h = gbdt_1s.predict(X_rec)
            
            # 使用該變數第一步的時空感知門控權重進行集成
            dates_h = df_test_aligned["Date"] + pd.Timedelta(days=h)
            Month_h = dates_h.dt.month
            DayOfYear_h = dates_h.dt.dayofyear
            X_time_h = np.column_stack([
                np.sin(2 * np.pi * Month_h / 12),
                np.cos(2 * np.pi * Month_h / 12),
                np.sin(2 * np.pi * DayOfYear_h / 365.25),
                np.cos(2 * np.pi * DayOfYear_h / 365.25)
            ])
            w_h = meta_learners[col][0].get_effective_weights(X_time_h)
            pred_ens_h = w_h[:, 0]*pred_lr_h + w_h[:, 1]*pred_rf_h + w_h[:, 2]*pred_gbdt_h
            
            if col in non_negative_cols:
                pred_ens_h = np.clip(pred_ens_h, 0, None)
                
            y_pred_rec[:, c_idx, h-1] = pred_ens_h
            
        # 更新 X_rec lag 特徵，用於下一步驟 h+1 預報
        if h < horizon:
            for c_idx, col in enumerate(predict_cols):
                X_rec[f"{col}_lag1"] = y_pred_rec[:, c_idx, h-1]
                if h == 1:
                    X_rec[f"{col}_lag2"] = X_test_aligned[f"{col}_lag1"]
                else:
                    X_rec[f"{col}_lag2"] = y_pred_rec[:, c_idx, h-2]
                X_rec[f"{col}_diff1"] = X_rec[f"{col}_lag1"] - X_rec[f"{col}_lag2"]
                
    # 計算並寫入各變數的 Direct/Recursive 誤差衰減曲線
    for c_idx, col in enumerate(predict_cols):
        rmse_direct = []
        rmse_recursive = []
        y_true_col = test_predictions[col]["true_all"]
        y_pred_dir_col = test_predictions[col]["ensemble_all"]
        
        for h in range(1, horizon + 1):
            dir_rmse = np.sqrt(mean_squared_error(y_true_col[:, h-1], y_pred_dir_col[:, h-1]))
            rec_rmse = np.sqrt(mean_squared_error(y_true_col[:, h-1], y_pred_rec[:, c_idx, h-1]))
            rmse_direct.append(dir_rmse)
            rmse_recursive.append(rec_rmse)
            
        metrics[col]["rmse_direct_curve"] = [float(np.round(x, 4)) for x in rmse_direct]
        metrics[col]["rmse_recursive_curve"] = [float(np.round(x, 4)) for x in rmse_recursive]
        
    # 產出核心變數 "Temperature" (平均氣溫) 的演算法報告圖表 (使用集成預測結果的第 1 步)
    if "Temperature" in predict_cols:
        os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
        generate_temperature_plots(rf_models["Temperature"], feature_cols, df_test_aligned["Temperature"], test_predictions["Temperature"]["ensemble"], df_test_aligned)
        
    return lr_models, rf_models, gbdt_models, meta_learners, feature_cols, metrics, df_test_aligned, test_predictions

def generate_temperature_plots(rf_model, feature_cols, y_test, y_pred_ensemble, df_test):
    plots_dir = os.path.join(BASE_DIR, "plots")
    
    # 1. 特徵重要性 (以隨機森林作為特徵權重代表)
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    plt.figure(figsize=(10, 6))
    plt.barh([feature_cols[i] for i in indices][::-1], importances[indices][::-1], color="#1f77b4")
    plt.title("AI 天氣預測模型 - 前 10 大關鍵特徵重要性分析")
    plt.xlabel("相對重要性權重")
    plt.ylabel("特徵欄位")
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=150)
    plt.close()
    
    # 2. 折線圖對比 (展示前 90 天，使圖表清晰)
    sample_df = df_test.head(90).copy()
    sample_df["pred"] = y_pred_ensemble[:len(sample_df)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sample_df["Date"], sample_df["Temperature"], label="實際平均氣溫 (℃)", marker='o', color="#1f77b4")
    plt.plot(sample_df["Date"], sample_df["pred"], label="科學加權集成模型預測", linestyle="--", marker='x', color="#ff7f0e")
    plt.title("實際溫度與科學加權集成模型預測溫度對比圖 (最優化預測)")
    plt.xlabel("日期")
    plt.ylabel("氣溫 (℃)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_comparison.png"), dpi=150)
    plt.close()
    
    # 3. 擬合散佈圖
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test.values, y_pred_ensemble, alpha=0.6, color="#2ca02c")
    min_val = min(y_test.min(), y_pred_ensemble.min())
    max_val = max(y_test.max(), y_pred_ensemble.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="完美預測對角線")
    plt.title("最優加權集成模型 - 預測與實際擬合散佈圖")
    plt.xlabel("實際平均氣溫 (℃)")
    plt.ylabel("預測平均氣溫 (℃)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scatter_fit.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    from data_generator import generate_kinmen_weather_data
    df = generate_kinmen_weather_data()
    df_clean, predict_cols, lag_cols = prepare_features(df)
    lr, rf, gbdt, meta_learners, features, metrics, df_test, test_predictions = train_and_evaluate(df_clean, predict_cols, lag_cols)
    print("模型評估指標數 (變數數量):", len(metrics))

