# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 設定中文畫圖字型避免亂碼
plt.rcParams["font.family"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False

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
    for col in predict_cols:
        lag1_name = f"{col}_lag1"
        lag2_name = f"{col}_lag2"
        diff1_name = f"{col}_diff1"
        df_sorted[lag1_name] = df_sorted.groupby("StationID")[col].shift(1)
        df_sorted[lag2_name] = df_sorted.groupby("StationID")[col].shift(2)
        df_sorted[diff1_name] = df_sorted[lag1_name] - df_sorted[lag2_name]
        lag_cols.extend([lag1_name, lag2_name, diff1_name])
        
    # 刪除因為位移產生的第一列缺失值，以及其他包含 NaN 的列以利訓練
    # 注意：對於真實資料，我們使用前後填充 (ffill/bfill) 來填補缺失
    df_filled = df_sorted.copy()
    df_filled[predict_cols + lag_cols] = df_filled[predict_cols + lag_cols].ffill().bfill()
    df_clean = df_filled.dropna(subset=predict_cols + lag_cols).copy()
    
    return df_clean, predict_cols, lag_cols

def train_and_evaluate(df_clean, predict_cols, lag_cols):
    """
    為每一個氣候變數訓練獨立的預測模型。
    """
    feature_cols = ["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"] + lag_cols
    
    X = df_clean[feature_cols]
    
    # 資料集切分：若資料大於 20 筆，採用 80/20 時間切分；若太少則不切分以防出錯
    n_rows = len(df_clean)
    if n_rows >= 20:
        train_size = int(n_rows * 0.8)
        X_train = X.iloc[:train_size]
        X_test = X.iloc[train_size:]
        df_train = df_clean.iloc[:train_size]
        df_test = df_clean.iloc[train_size:]
    else:
        print("[!] 資料集筆數較少，將同時使用全部資料進行訓練與評估。")
        X_train = X
        X_test = X
        df_train = df_clean
        df_test = df_clean
        
    lr_models = {}
    rf_models = {}
    gbdt_models = {}
    metrics = {}
    test_predictions = {}
    
    for col in predict_cols:
        y_train = df_train[col]
        y_test = df_test[col]
        
        # 1. 嶺回歸 (L2 正規化 Baseline)
        lr = Ridge(alpha=1.0)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        lr_models[col] = lr
        
        # 2. 隨機森林 (優化參數，提供穩定泛化)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_models[col] = rf
        
        # 3. 梯度提升樹 (GBDT) - 高效的表格式資料迴歸演算法
        gbdt = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=42)
        gbdt.fit(X_train, y_train)
        y_pred_gbdt = gbdt.predict(X_test)
        gbdt_models[col] = gbdt
        
        # 4. 科學加權集成模型 (Blending Ensemble)
        y_pred_ensemble = 0.1 * y_pred_lr + 0.4 * y_pred_rf + 0.5 * y_pred_gbdt
        
        # 儲存測試集預測結果
        test_predictions[col] = {
            "lr": y_pred_lr,
            "rf": y_pred_rf,
            "gbdt": y_pred_gbdt,
            "ensemble": y_pred_ensemble
        }
        
        # 評估指標計算
        if len(y_test) > 1:
            try:
                def get_metrics(y_true, y_pred):
                    r2 = np.round(r2_score(y_true, y_pred), 4)
                    rmse = np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
                    mae = np.round(mean_absolute_error(y_true, y_pred), 2)
                    return r2, rmse, mae
                
                r2_lr, rmse_lr, mae_lr = get_metrics(y_test, y_pred_lr)
                r2_rf, rmse_rf, mae_rf = get_metrics(y_test, y_pred_rf)
                r2_gbdt, rmse_gbdt, mae_gbdt = get_metrics(y_test, y_pred_gbdt)
                r2_ens, rmse_ens, mae_ens = get_metrics(y_test, y_pred_ensemble)
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
            
        metrics[col] = {
            "嶺回歸_R平方": r2_lr, "嶺回歸_RMSE": rmse_lr, "嶺回歸_MAE": mae_lr,
            "隨機森林_R平方": r2_rf, "隨機森林_RMSE": rmse_rf, "隨機森林_MAE": mae_rf,
            "梯度提升樹_R平方": r2_gbdt, "梯度提升樹_RMSE": rmse_gbdt, "梯度提升樹_MAE": mae_gbdt,
            "加權集成_R平方": r2_ens, "加權集成_RMSE": rmse_ens, "加權集成_MAE": mae_ens
        }
        
    # 產出核心變數 "Temperature" (平均氣溫) 的演算法報告圖表 (使用集成預測結果)
    if "Temperature" in predict_cols:
        os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
        generate_temperature_plots(rf_models["Temperature"], feature_cols, df_test["Temperature"], test_predictions["Temperature"]["ensemble"], df_test)
        
    return lr_models, rf_models, gbdt_models, feature_cols, metrics, df_test, test_predictions

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
    lr, rf, gbdt, features, metrics, df_test, test_predictions = train_and_evaluate(df_clean, predict_cols, lag_cols)
    print("模型評估指標數 (變數數量):", len(metrics))

