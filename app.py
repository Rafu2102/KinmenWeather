# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from data_generator import generate_kinmen_weather_data
from model import prepare_features, train_and_evaluate

# 設定網頁配置
st.set_page_config(
    page_title="金門氣候特徵分析與機器學習預估系統",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 解決 Matplotlib 中文顯示問題 (相容 Windows 與 Linux 雲端環境)
plt.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 中央氣象署 (CWA) 欄位之正統繁體中文對照表
COLUMN_CN_MAP = {
    "StnPres": "測站平均氣壓 (hPa)",
    "SeaPres": "海平面平均氣壓 (hPa)",
    "StnPresMax": "最高氣壓 (hPa)",
    "StnPresMaxTime": "最高氣壓發生時間",
    "StnPresMin": "最低氣壓 (hPa)",
    "StnPresMinTime": "最低氣壓發生時間",
    "Temperature": "平均氣溫 (℃)",
    "T Max": "最高氣溫 (℃)",
    "T Max Time": "最高氣溫發生時間",
    "T Min": "最低氣溫 (℃)",
    "T Min Time": "最低氣溫發生時間",
    "Td dew point": "平均露點溫度 (℃)",
    "RH": "平均相對濕度 (%)",
    "RHMin": "最小相對濕度 (%)",
    "RHMinTime": "最小相對濕度發生時間",
    "WS": "平均風速 (m/s)",
    "WD": "平均風向 (度)",
    "WSGust": "最大陣風風速 (m/s)",
    "WDGust": "最大陣風風向 (度)",
    "WGustTime": "最大陣風發生時間",
    "Precp": "累積降水量 (mm)",
    "PrecpHour": "降水時數 (小時)",
    "PrecpMax10": "最大 10 分鐘降水量 (mm)",
    "PrecpMax10Time": "最大 10 分鐘降水時間",
    "PrecpMax60": "最大 60 分鐘降水量 (mm)",
    "PrecpMax60Time": "最大 60 分鐘降水時間",
    "SunShine": "日照時數 (小時)",
    "SunshineRate": "日照率 (%)",
    "GloblRad": "全天太陽輻射 (MJ/㎡)",
    "VisbMean": "平均能見度 (公里)",
    "EvapA": "蒸發量 (mm)",
    "UVI Max": "最大紫外線指數",
    "UVI Max Time": "最大紫外線指數時間",
    "Cloud Amount": "平均總雲量 (0-10)",
    "TxSoil0cm": "地表 0cm 溫度 (℃)",
    "TxSoil5cm": "地中 5cm 溫度 (℃)",
    "TxSoil10cm": "地中 10cm 溫度 (℃)",
    "TxSoil20cm": "地中 20cm 溫度 (℃)",
    "TxSoil30cm": "地中 30cm 溫度 (℃)",
    "TxSoil50cm": "地中 50cm 溫度 (℃)",
    "TxSoil100cm": "地中 100cm 溫度 (℃)",
    "Cloud Amount Sat": "衛星觀測總雲量 (0-10)",
    "VisbMean Auto": "自動觀測平均能見度 (公里)"
}

CN_COLUMN_MAP = {v: k for k, v in COLUMN_CN_MAP.items()}

# 物理上不可能為負數的氣象項目
NON_NEGATIVE_COLS = [
    "Precp", "PrecpHour", "PrecpMax10", "PrecpMax60", 
    "WS", "WSGust", "SunShine", "SunshineRate", 
    "GloblRad", "EvapA", "UVI Max"
]

# 快取資料載入以加速運行
@st.cache_data
def load_and_preprocess_data():
    df = generate_kinmen_weather_data()
    # 關鍵修正：將 StationID 強制轉為字串格式，避免 CSV 讀取時將測站識別碼誤判為 int64
    # 這能確保與側邊欄選單字串匹配，解決歷史資料篩選為空的問題
    df["StationID"] = df["StationID"].astype(str)
    df_clean, predict_cols, lag_cols = prepare_features(df)
    df_clean["StationID"] = df_clean["StationID"].astype(str)
    return df, df_clean, predict_cols, lag_cols

# 快取模型訓練
@st.cache_resource
def train_models_cached(df_clean, predict_cols, lag_cols):
    lr_models, rf_models, gbdt_models, feature_names, metrics_dict, df_test, test_predictions = train_and_evaluate(df_clean, predict_cols, lag_cols)
    return lr_models, rf_models, gbdt_models, feature_names, metrics_dict, df_test, test_predictions

# 載入資料與模型
try:
    df, df_clean, predict_cols, lag_cols = load_and_preprocess_data()
    lr_models, rf_models, gbdt_models, feature_names, metrics_dict, df_test, test_predictions = train_models_cached(df_clean, predict_cols, lag_cols)
    load_success = True
except Exception as e:
    load_success = False
    load_error_msg = str(e)

# 網頁標題
st.title("⚡ 金門氣候 CWA 觀測數據分析與機器學習預估儀表板")
st.markdown("本系統採用**中央氣象署 (CWA) 官方格式**之觀測資料，對金門測站的所有數值氣象欄位進行機器學習建模與氣候推估（100% 真實數據）。")

if not load_success:
    st.error("🚨 載入氣象資料集時出錯！")
    st.info(
        f"**詳細錯誤原因**：{load_error_msg}\n\n"
        "**建議解決步驟**：\n"
        "1. 請檢查 `金門天氣預測專案/日月CSV/` 資料夾中是否已放置氣象署下載的每日觀測資料 CSV 檔。\n"
        "2. 確保 CSV 檔案命名符合格式：`[測站ID]-[年份]-[月份].csv` (例如: `467110-2026-06.csv`)。\n"
        "3. 確保檔案內容第一列為中文標頭，第二列為英文標頭（即氣象署官方原始下載格式）。"
    )
else:
    # 測站顯示名稱中文化
    station_labels = {
        "467110": "467110 (金門氣象站 - 金城鎮)"
    }
    
    # 側邊欄控制面板
    st.sidebar.header("⚙️ 測站與時間篩選")

    # 區域選擇
    station_options = df["StationID"].unique()
    station_display = [station_labels.get(st_id, f"{st_id} (觀測站)") for st_id in station_options]
    selected_display = st.sidebar.selectbox(
        "選擇氣象觀測站",
        options=station_display,
        index=0
    )
    selected_station = selected_display.split(" ")[0]

    # 時間區段選擇
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()

    if min_date == max_date:
        st.sidebar.warning(f"目前資料庫中僅有 1 日的觀測紀錄：{min_date.strftime('%Y/%m/%d')}")
        start_date, end_date = min_date, max_date
    else:
        start_date, end_date = st.sidebar.slider(
            "選擇歷史資料探索時間區段",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY/MM/DD"
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **系統操作說明**：\n"
        "1. 本系統已將氣壓、氣溫、相對濕度、累積降水量、日照時數、平均風速與地表土壤溫度等所有數值欄位整合建模。\n"
        "2. 前往「AI 天氣預測」分頁，選擇任意日期（支援未來日期），機器學習演算法將自動提供該預估日的完整天氣指標預報。\n"
        "3. 對預估原理有興趣者，可展開情境模擬控制台微調昨日天氣因子，進行 What-if 敏感度分析。"
    )

    # 將英文欄位轉換為繁體中文清單
    predict_cols_cn = [COLUMN_CN_MAP.get(col, col) for col in predict_cols]

    # 建立分頁
    tab1, tab2, tab3 = st.tabs(["📊 歷史數據探索", "🔮 AI 天氣預報與推估", "⚙️ 演算法成效分析"])

    # ==================== TAB 1: 歷史數據探索 ====================
    with tab1:
        st.header("📊 歷史天氣指標趨勢與多維度特徵分析")
        
        # 篩選資料
        mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date)) & (df["StationID"] == selected_station)
        filtered_df = df[mask]
        
        # 顯示關鍵指標卡片
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            val = filtered_df['Temperature'].mean() if 'Temperature' in filtered_df.columns else np.nan
            st.metric("平均氣溫", f"{val:.1f} °C" if not np.isnan(val) else "N/A")
        with col2:
            val = filtered_df['T Max'].max() if 'T Max' in filtered_df.columns else np.nan
            st.metric("最高氣溫", f"{val:.1f} °C" if not np.isnan(val) else "N/A")
        with col3:
            val = filtered_df['T Min'].min() if 'T Min' in filtered_df.columns else np.nan
            st.metric("最低氣溫", f"{val:.1f} °C" if not np.isnan(val) else "N/A")
        with col4:
            val = filtered_df['Precp'].sum() if 'Precp' in filtered_df.columns else np.nan
            st.metric("累積降雨量", f"{val:.1f} mm" if not np.isnan(val) else "N/A")
            
        st.markdown("---")
        
        # 選擇畫圖變數（選單選項中文化）
        selected_cn = st.selectbox(
            "選擇要繪製歷史趨勢與統計分析的氣象觀測指標",
            options=predict_cols_cn,
            index=predict_cols_cn.index(COLUMN_CN_MAP["Temperature"]) if "Temperature" in predict_cols else 0
        )
        plot_var = CN_COLUMN_MAP.get(selected_cn, selected_cn)
        
        if len(filtered_df) == 0:
            st.warning("⚠️ 所選時間範圍內無對應觀測資料。")
        else:
            # 1. 歷史變化折線圖 (加 30 天滾動平均)
            st.subheader(f"📈 觀測指標: {selected_cn} 歷史變化趨勢圖 (附 30 日滾動平均)")
            fig, ax = plt.subplots(figsize=(13, 4.8))
            ax.plot(filtered_df["Date"], filtered_df[plot_var], label="每日實際觀測值", color="#1f77b4", alpha=0.4)
            # 計算 30 天滾動平均
            if len(filtered_df) >= 30:
                ma_30 = filtered_df[plot_var].rolling(window=30, min_periods=1).mean()
                ax.plot(filtered_df["Date"], ma_30, label="30 天滾動平均線", color="#ff7f0e", linewidth=2.5)
            ax.set_xlabel("日期", fontsize=9)
            ax.set_ylabel(selected_cn, fontsize=9)
            ax.tick_params(axis='both', labelsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            st.markdown("---")
            
            # 2. 直方圖 (KDE) 與 箱線圖 併排
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.subheader("📊 數據分佈直方圖與機率密度 (Distribution & Density)")
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                clean_series = filtered_df[plot_var].dropna()
                if len(clean_series) > 0:
                    counts, bins, patches = ax2.hist(clean_series, bins=25, density=True, alpha=0.6, color="#2ca02c", edgecolor='grey', label="觀測值頻率")
                    # KDE 密度估計
                    try:
                        std_val = clean_series.std()
                        if len(clean_series) > 1 and std_val > 0:
                            from scipy.stats import gaussian_kde
                            kde = gaussian_kde(clean_series)
                            x_grid = np.linspace(clean_series.min(), clean_series.max(), 200)
                            ax2.plot(x_grid, kde(x_grid), color="#d62728", linewidth=2, label="KDE 密度曲線")
                        else:
                            raise ValueError("無法進行 KDE 計算")
                    except Exception:
                        mu, std = clean_series.mean(), clean_series.std()
                        if std > 0:
                            x_grid = np.linspace(clean_series.min(), clean_series.max(), 200)
                            p = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_grid - mu) / std) ** 2)
                            ax2.plot(x_grid, p, color="#d62728", linewidth=2, label="常態分佈擬合線")
                    ax2.set_xlabel(selected_cn, fontsize=9)
                    ax2.set_ylabel("機率密度", fontsize=9)
                    ax2.tick_params(axis='both', labelsize=9)
                    ax2.grid(True, linestyle=":", alpha=0.6)
                    ax2.legend(fontsize=9)
                else:
                    ax2.text(0.5, 0.5, "無足夠數據繪製分佈圖", ha='center', va='center')
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
                
            with col_t2:
                st.subheader("📅 月份季節性箱線圖 (Monthly Boxplot)")
                fig3, ax3 = plt.subplots(figsize=(7, 5))
                box_df = filtered_df.copy()
                box_df["Month"] = box_df["Date"].dt.month
                months = sorted(box_df["Month"].unique())
                if len(months) > 0:
                    data_groups = [box_df[box_df["Month"] == m][plot_var].dropna().values for m in months]
                    ax3.boxplot(data_groups, tick_labels=[f"{m}月" for m in months], patch_artist=True,
                                boxprops=dict(facecolor="#17becf", color="#1a1a1a", alpha=0.8),
                                medianprops=dict(color="red", linewidth=1.5))
                    ax3.set_xlabel("月份", fontsize=9)
                    ax3.set_ylabel(selected_cn, fontsize=9)
                    ax3.tick_params(axis='both', labelsize=9)
                    ax3.grid(True, linestyle=":", alpha=0.4)
                else:
                    ax3.text(0.5, 0.5, "無足夠數據繪製季節箱線圖", ha='center', va='center')
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
                
            st.markdown("---")
            
            # 3. 氣候變數相關性熱力圖 (軸標籤中文化)
            st.subheader("🔗 關鍵氣候指標相關性熱力圖 (Climate Correlation Heatmap)")
            key_vars = ["Temperature", "T Max", "T Min", "Precp", "RH", "WS", "WD", "StnPres", "Evap", "Tx LST"]
            existing_vars = [v for v in key_vars if v in filtered_df.columns]
            
            if len(existing_vars) > 1:
                corr_matrix = filtered_df[existing_vars].corr()
                fig4, ax4 = plt.subplots(figsize=(11, 8.5))
                cax = ax4.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
                fig4.colorbar(cax)
                
                # 轉成中文標籤
                existing_vars_cn = [COLUMN_CN_MAP.get(v, v) for v in existing_vars]
                
                ax4.set_xticks(np.arange(len(existing_vars)))
                ax4.set_yticks(np.arange(len(existing_vars)))
                ax4.set_xticklabels(existing_vars_cn, rotation=45, ha='right', fontsize=9)
                ax4.set_yticklabels(existing_vars_cn, fontsize=9)
                
                for i in range(len(existing_vars)):
                    for j in range(len(existing_vars)):
                        val = corr_matrix.iloc[i, j]
                        ax4.text(j, i, f"{val:.2f}", ha='center', va='center', 
                                 color="white" if abs(val) > 0.45 else "black", fontsize=8)
                                 
                ax4.set_title("關鍵氣候變數 Pearson 相關係數矩陣", fontsize=11, pad=15)
                plt.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.info("無足夠氣候變數繪製相關性熱力圖。")

    # ==================== TAB 2: AI 天氣預測 ====================
    with tab2:
        st.header("🔮 AI 全欄位氣象預估")
        st.markdown(
            "系統已實作**遞迴自迴歸預測演算法 (Recursive Autoregressive Forecasting)**。當您選擇未來的任意日期時，"
            "模型會自動以昨日的預測結果反饋至今日的輸入特徵，逐步向後滾動推算。**您可以直接獲得該預報日期的所有完整氣候指標值，無需手動調整滑桿。**"
        )
        
        # 允許選擇未來 10 年內的日期進行預測與情境模擬
        predict_date = st.date_input(
            "選擇預估目標日期 (支援歷史與未來日期)",
            value=pd.to_datetime(max_date).to_pydatetime(),
            min_value=pd.to_datetime(min_date).to_pydatetime(),
            max_value=(pd.to_datetime(max_date) + pd.Timedelta(days=365 * 10)).to_pydatetime()
        )
        
        # 判斷是否為未來日期
        is_future = pd.to_datetime(predict_date) > pd.to_datetime(max_date)
        
        # 執行遞迴預測邏輯
        # 找出最後一個已知的日期
        station_rows = df_clean[df_clean["StationID"] == selected_station].sort_values(by="Date")
        if len(station_rows) == 0:
            station_rows = df_clean.sort_values(by="Date")
        
        max_known_row = station_rows.iloc[-1]
        max_known_date = max_known_row["Date"]
        
        # 初始化 st.session_state 增量預估快取
        if "forecast_cache" not in st.session_state or st.session_state.get("cache_station") != selected_station:
            init_base_row = {
                "Date": pd.to_datetime(max_known_date),
                "StationID": selected_station,
                "Month": max_known_date.month,
                "DayOfYear": max_known_date.dayofyear,
                "Month_sin": np.sin(2 * np.pi * max_known_date.month / 12),
                "Month_cos": np.cos(2 * np.pi * max_known_date.month / 12),
                "DayOfYear_sin": np.sin(2 * np.pi * max_known_date.dayofyear / 365.25),
                "DayOfYear_cos": np.cos(2 * np.pi * max_known_date.dayofyear / 365.25)
            }
            for k in max_known_row.index:
                init_base_row[k] = max_known_row[k]
                
            st.session_state.forecast_cache = {
                pd.to_datetime(max_known_date): {
                    "base_row": init_base_row,
                    "pred_rf_dict": {col: max_known_row[col] for col in predict_cols},
                    "pred_lr_dict": {col: max_known_row[col] for col in predict_cols},
                    "pred_gbdt_dict": {col: max_known_row[col] for col in predict_cols},
                    "pred_ensemble_dict": {col: max_known_row[col] for col in predict_cols}
                }
            }
            st.session_state.cache_station = selected_station
            
        # 計算日期天數差
        days_diff = (pd.to_datetime(predict_date) - max_known_date).days
        
        # 建立虛擬 base_row 容器，並初始化所有基礎日期特徵與週期性轉換特徵
        pred_month = predict_date.month
        pred_dayofyear = pd.to_datetime(predict_date).dayofyear
        
        base_row = {
            "Date": pd.to_datetime(predict_date),
            "StationID": selected_station,
            "Month": pred_month,
            "DayOfYear": pred_dayofyear,
            "Month_sin": np.sin(2 * np.pi * pred_month / 12),
            "Month_cos": np.cos(2 * np.pi * pred_month / 12),
            "DayOfYear_sin": np.sin(2 * np.pi * pred_dayofyear / 365.25),
            "DayOfYear_cos": np.cos(2 * np.pi * pred_dayofyear / 365.25)
        }
        
        target_timestamp = pd.to_datetime(predict_date)
        
        if days_diff <= 0:
            # 歷史日期：直接提取
            target_row = df_clean[
                (df_clean["Date"] == pd.to_datetime(predict_date)) & 
                (df_clean["StationID"] == selected_station)
            ]
            if len(target_row) > 0:
                base_row_extracted = target_row.iloc[0]
            else:
                # 歷史中可能缺失的日期，尋找最接近的一天
                time_diff = (df_clean["Date"] - pd.to_datetime(predict_date)).abs()
                base_row_extracted = df_clean.loc[time_diff.idxmin()]
            
            # 將提取出來的所有特徵深拷貝寫入 base_row 容器，確保 What-if 控制端取得完整時間週期欄位
            for k in base_row_extracted.index:
                base_row[k] = base_row_extracted[k]
                
            adjusted_features = {}
            for f in feature_names:
                adjusted_features[f] = base_row[f]
                
            X_pred = pd.DataFrame([adjusted_features])[feature_names]
            
            pred_rf_dict = {}
            pred_lr_dict = {}
            pred_gbdt_dict = {}
            pred_ensemble_dict = {}
            for col in predict_cols:
                pred_rf_dict[col] = rf_models[col].predict(X_pred)[0]
                pred_lr_dict[col] = lr_models[col].predict(X_pred)[0]
                pred_gbdt_dict[col] = gbdt_models[col].predict(X_pred)[0]
                if col in NON_NEGATIVE_COLS:
                    pred_rf_dict[col] = max(0.0, pred_rf_dict[col])
                    pred_lr_dict[col] = max(0.0, pred_lr_dict[col])
                    pred_gbdt_dict[col] = max(0.0, pred_gbdt_dict[col])
                w_lr = metrics_dict[col].get("權重_Ridge", 0.1)
                w_rf = metrics_dict[col].get("權重_RF", 0.4)
                w_gbdt = metrics_dict[col].get("權重_GBDT", 0.5)
                pred_ensemble_dict[col] = w_lr * pred_lr_dict[col] + w_rf * pred_rf_dict[col] + w_gbdt * pred_gbdt_dict[col]
        else:
            # 未來日期：優先從 st.session_state 快取中增量讀取或推算
            if target_timestamp in st.session_state.forecast_cache:
                # 情況 1：目標日期已被計算過並存在於快取中，直接讀取 (耗時 0 毫秒)
                cached_data = st.session_state.forecast_cache[target_timestamp]
                base_row = dict(cached_data["base_row"])
                pred_rf_dict = dict(cached_data["pred_rf_dict"])
                pred_lr_dict = dict(cached_data["pred_lr_dict"])
                pred_gbdt_dict = dict(cached_data["pred_gbdt_dict"])
                pred_ensemble_dict = dict(cached_data["pred_ensemble_dict"])
            else:
                # 情況 2：目標日期不在快取中，找出快取中所有小於目標日期的最接近日期，進行最小步數增量預測
                cached_dates = [d for d in st.session_state.forecast_cache.keys() if d < target_timestamp]
                last_cached_date = max(cached_dates) if len(cached_dates) > 0 else pd.to_datetime(max_known_date)
                
                start_cached = st.session_state.forecast_cache[last_cached_date]
                current_lag1 = dict(start_cached["pred_ensemble_dict"])
                
                current_lag2 = {}
                for col in predict_cols:
                    if last_cached_date == pd.to_datetime(max_known_date):
                        current_lag2[col] = max_known_row[f"{col}_lag1"]
                    else:
                        prev_date = last_cached_date - pd.Timedelta(days=1)
                        if prev_date in st.session_state.forecast_cache:
                            current_lag2[col] = st.session_state.forecast_cache[prev_date]["pred_ensemble_dict"][col]
                        else:
                            current_lag2[col] = start_cached["base_row"][f"{col}_lag2"]
                
                # 計算增量推算天數
                inc_days = (target_timestamp - last_cached_date).days
                
                # 使用 NumPy 加速：建立欄位索引對應，避免在迴圈內頻繁建構 DataFrame 與重排欄位之巨大開銷
                feat_idx = {name: idx for idx, name in enumerate(feature_names)}
                x_pred_arr = np.zeros((1, len(feature_names)))
                
                # 建立進度條
                progress_bar = None
                if inc_days > 90:
                    progress_bar = st.progress(0.0, text="正在增量推估氣候指標...")
                    
                for d in range(1, inc_days + 1):
                    curr_date = last_cached_date + pd.Timedelta(days=d)
                    curr_month = curr_date.month
                    curr_dayofyear = curr_date.dayofyear
                    
                    # 建立當日的 base_row 容器
                    curr_base_row = {
                        "Date": curr_date,
                        "StationID": selected_station,
                        "Month": curr_month,
                        "DayOfYear": curr_dayofyear,
                        "Month_sin": np.sin(2 * np.pi * curr_month / 12),
                        "Month_cos": np.cos(2 * np.pi * curr_month / 12),
                        "DayOfYear_sin": np.sin(2 * np.pi * curr_dayofyear / 365.25),
                        "DayOfYear_cos": np.cos(2 * np.pi * curr_dayofyear / 365.25)
                    }
                    
                    # 直接寫入預先定義好的 NumPy 陣列以降低記憶體配置開銷
                    x_pred_arr[0, feat_idx["Month_sin"]] = curr_base_row["Month_sin"]
                    x_pred_arr[0, feat_idx["Month_cos"]] = curr_base_row["Month_cos"]
                    x_pred_arr[0, feat_idx["DayOfYear_sin"]] = curr_base_row["DayOfYear_sin"]
                    x_pred_arr[0, feat_idx["DayOfYear_cos"]] = curr_base_row["DayOfYear_cos"]
                    
                    for col in predict_cols:
                        x_pred_arr[0, feat_idx[f"{col}_lag1"]] = current_lag1[col]
                        x_pred_arr[0, feat_idx[f"{col}_lag2"]] = current_lag2[col]
                        x_pred_arr[0, feat_idx[f"{col}_diff1"]] = current_lag1[col] - current_lag2[col]
                        
                        # 備份 lag 特徵至 base_row 以免敏感度微調產生 KeyError
                        curr_base_row[f"{col}_lag1"] = current_lag1[col]
                        curr_base_row[f"{col}_lag2"] = current_lag2[col]
                        curr_base_row[f"{col}_diff1"] = current_lag1[col] - current_lag2[col]
                    
                    # 一次性包裝成 DataFrame，避免 sklearn 拋出特徵名稱警告，且只做行名封裝不重排
                    X_pred = pd.DataFrame(x_pred_arr, columns=feature_names)
                    
                    step_rf = {}
                    step_lr = {}
                    step_gbdt = {}
                    step_ensemble = {}
                    for col in predict_cols:
                        step_rf[col] = rf_models[col].predict(X_pred)[0]
                        step_lr[col] = lr_models[col].predict(X_pred)[0]
                        step_gbdt[col] = gbdt_models[col].predict(X_pred)[0]
                        if col in NON_NEGATIVE_COLS:
                            step_rf[col] = max(0.0, step_rf[col])
                            step_lr[col] = max(0.0, step_lr[col])
                            step_gbdt[col] = max(0.0, step_gbdt[col])
                        w_lr = metrics_dict[col].get("權重_Ridge", 0.1)
                        w_rf = metrics_dict[col].get("權重_RF", 0.4)
                        w_gbdt = metrics_dict[col].get("權重_GBDT", 0.5)
                        step_ensemble[col] = w_lr * step_lr[col] + w_rf * step_rf[col] + w_gbdt * step_gbdt[col]
                    
                    # 寫入預估值
                    for col in predict_cols:
                        curr_base_row[col] = step_ensemble[col]
                        
                    # 存入 st.session_state 快取
                    st.session_state.forecast_cache[curr_date] = {
                        "base_row": curr_base_row,
                        "pred_rf_dict": step_rf,
                        "pred_lr_dict": step_lr,
                        "pred_gbdt_dict": step_gbdt,
                        "pred_ensemble_dict": step_ensemble
                    }
                    
                    # 滾動更新 Lag
                    for col in predict_cols:
                        current_lag2[col] = current_lag1[col]
                        current_lag1[col] = step_ensemble[col]
                    
                    # 更新進度條 (每前進 5% 更新一次)
                    if progress_bar and d % max(1, inc_days // 20) == 0:
                        progress_bar.progress(d / inc_days, text=f"正在增量推估中... 已計算至第 {d}/{inc_days} 天")
                
                if progress_bar:
                    progress_bar.empty()
                
                # 計算完畢，提取目標結果
                cached_data = st.session_state.forecast_cache[target_timestamp]
                base_row = dict(cached_data["base_row"])
                pred_rf_dict = dict(cached_data["pred_rf_dict"])
                pred_lr_dict = dict(cached_data["pred_lr_dict"])
                pred_gbdt_dict = dict(cached_data["pred_gbdt_dict"])
                pred_ensemble_dict = dict(cached_data["pred_ensemble_dict"])
            
            st.info(
                f"ℹ️ 您選擇了未來日期 {predict_date.strftime('%Y/%m/%d')}。系統已啟動遞迴推算，"
                f"使用最新觀測日 ({max_known_date.strftime('%Y/%m/%d')}) 遞迴演算法預測 {days_diff} 天後的氣候指標。"
            )
            if days_diff > 365:
                st.warning(
                    "⚠️ **學術提示：超長週期自迴歸預估特徵**\n\n"
                    "由於預估目標日期離已知觀測日已超過 1 年，在遞迴自迴歸 (Autoregressive) 預測中，"
                    "隨著推算步數增加，預測值將逐漸消除短期隨機波動，並受月份與日曆週期特徵 (Month_sin/cos, DayOfYear_sin/cos) 主導，"
                    "最後會穩定收斂至歷史的統計平均波動規律 (氣候態)。這是時間序列長期推估的正常現象。"
                )
            
        # 可折疊的微調控制台 (情境模擬)
        with st.expander("🎛️ 氣候因子情境模擬控制台 (選用 - 提供氣候變量微調分析)"):
            st.markdown("當您想要模擬「如果昨天的天氣產生變化，今天預報會如何改變」的情境時，可利用下方滑桿進行微調：")
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            
            sim_features = {}
            for f in feature_names:
                sim_features[f] = base_row[f]
                
            with col_ctrl1:
                if "Temperature_lag1" in sim_features:
                    val = st.slider("昨日平均氣溫 (℃)", min_value=0.0, max_value=40.0, value=float(base_row["Temperature_lag1"]), step=0.5)
                    sim_features["Temperature_lag1"] = val
                    if "Temperature_lag2" in sim_features and "Temperature_diff1" in sim_features:
                        sim_features["Temperature_diff1"] = val - sim_features["Temperature_lag2"]
            with col_ctrl2:
                if "RH_lag1" in sim_features:
                    val = st.slider("昨日相對濕度 (%)", min_value=20.0, max_value=100.0, value=float(base_row["RH_lag1"]), step=1.0)
                    sim_features["RH_lag1"] = val
                    if "RH_lag2" in sim_features and "RH_diff1" in sim_features:
                        sim_features["RH_diff1"] = val - sim_features["RH_lag2"]
            with col_ctrl3:
                if "WS_lag1" in sim_features:
                    val = st.slider("昨日風速 (m/s)", min_value=0.0, max_value=30.0, value=float(base_row["WS_lag1"]), step=0.5)
                    sim_features["WS_lag1"] = val
                    if "WS_lag2" in sim_features and "WS_diff1" in sim_features:
                        sim_features["WS_diff1"] = val - sim_features["WS_lag2"]
            
            # 若有微調，則重新計算預估
            X_sim = pd.DataFrame([sim_features])[feature_names]
            for col in predict_cols:
                pred_rf_dict[col] = rf_models[col].predict(X_sim)[0]
                pred_lr_dict[col] = lr_models[col].predict(X_sim)[0]
                pred_gbdt_dict[col] = gbdt_models[col].predict(X_sim)[0]
                if col in NON_NEGATIVE_COLS:
                    pred_rf_dict[col] = max(0.0, pred_rf_dict[col])
                    pred_lr_dict[col] = max(0.0, pred_lr_dict[col])
                    pred_gbdt_dict[col] = max(0.0, pred_gbdt_dict[col])
                w_lr = metrics_dict[col].get("權重_Ridge", 0.1)
                w_rf = metrics_dict[col].get("權重_RF", 0.4)
                w_gbdt = metrics_dict[col].get("權重_GBDT", 0.5)
                pred_ensemble_dict[col] = w_lr * pred_lr_dict[col] + w_rf * pred_rf_dict[col] + w_gbdt * pred_gbdt_dict[col]
                
        st.markdown("---")
        st.subheader("🔮 今日氣象項目預估結果")
        
        # 1. 核心指標展示 (顯示加權集成結果，中文化對照)
        major_display_cols = ["Temperature", "T Max", "T Min", "Precp"]
        col_grid = st.columns(4)
        grid_idx = 0
        for col in predict_cols:
            if col in major_display_cols:
                cn_name = COLUMN_CN_MAP.get(col, col)
                with col_grid[grid_idx % 4]:
                    if is_future:
                        st.metric(
                            label=f"預估 {cn_name}",
                            value=f"{pred_ensemble_dict[col]:.1f}",
                            delta=f"Ridge預測: {pred_lr_dict[col]:.1f}"
                        )
                    else:
                        actual_val = base_row[col]
                        st.metric(
                            label=f"預估 {cn_name}",
                            value=f"{pred_ensemble_dict[col]:.1f}",
                            delta=f"實際: {actual_val:.1f} | Ridge: {pred_lr_dict[col]:.1f}"
                        )
                    grid_idx += 1
        
        # 2. 所有氣候欄位預報表格對照
        st.markdown("#### 📋 所有 CWA 氣候欄位預估值對照表")
        
        pred_table_rows = []
        for col in predict_cols:
            cn_name = COLUMN_CN_MAP.get(col, col)
            row_data = {
                "氣象項目 (CWA 欄位名稱)": cn_name,
                "昨日實際值": np.round(base_row[f"{col}_lag1"], 2),
                "加權集成預估 (Blending)": np.round(pred_ensemble_dict[col], 2),
                "隨機森林預估 (Bagging)": np.round(pred_rf_dict[col], 2),
                "梯度提升樹預估 (GBDT)": np.round(pred_gbdt_dict[col], 2),
                "嶺回歸預估 (Ridge)": np.round(pred_lr_dict[col], 2)
            }
            if is_future:
                row_data["今日實際值"] = "尚未觀測"
                row_data["預測誤差"] = "無法計算"
            else:
                row_data["今日實際值"] = np.round(base_row[col], 2)
                row_data["預測誤差"] = np.round(pred_ensemble_dict[col] - base_row[col], 2)
            pred_table_rows.append(row_data)
        
        st.dataframe(pd.DataFrame(pred_table_rows), width="stretch")

    # ==================== TAB 3: 演算法成效分析 ====================
    with tab3:
        st.header("⚙️ 演算法成效與決策特徵分析")
        st.markdown("本頁面展示機器學習演算法針對各觀測變數計算之評估指標，提供您期末報告必備的模型效能數據。")
        
        # 選擇評估變數（選單選項中文化）
        selected_analysis_cn = st.selectbox(
            "選擇要檢視模型指標與科學評估圖表的氣象觀測欄位",
            options=predict_cols_cn,
            index=predict_cols_cn.index(COLUMN_CN_MAP["Temperature"]) if "Temperature" in predict_cols else 0
        )
        analysis_var = CN_COLUMN_MAP.get(selected_analysis_cn, selected_analysis_cn)
        
        var_metrics = metrics_dict[analysis_var]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"#### 📊 {selected_analysis_cn} 模型成效指標對照表")
            comparison_table = pd.DataFrame({
                "評估指標名稱": ["R平方 (R-squared)", "均方根誤差 (RMSE)", "平均絕對誤差 (MAE)"],
                "嶺回歸 (Ridge)": [var_metrics["嶺回歸_R平方"], var_metrics["嶺回歸_RMSE"], var_metrics["嶺回歸_MAE"]],
                "隨機森林 (Bagging)": [var_metrics["隨機森林_R平方"], var_metrics["隨機森林_RMSE"], var_metrics["隨機森林_MAE"]],
                "梯度提升樹 (Boosting)": [var_metrics["梯度提升樹_R平方"], var_metrics["梯度提升樹_RMSE"], var_metrics["梯度提升樹_MAE"]],
                "加權集成模型 (Blending)": [var_metrics["加權集成_R平方"], var_metrics["加權集成_RMSE"], var_metrics["加權集成_MAE"]]
            })
            st.table(comparison_table)
            
            st.markdown("#### 🎯 Stacking 堆疊集成最優權重分配")
            w_lr = var_metrics.get("權重_Ridge", 0.1)
            w_rf = var_metrics.get("權重_RF", 0.4)
            w_gbdt = var_metrics.get("權重_GBDT", 0.5)
            weight_table = pd.DataFrame({
                "基準模型名稱": ["嶺回歸 (Ridge)", "隨機森林 (Random Forest)", "梯度提升樹 (GBDT)"],
                "演算法自動學習權重": [f"{w_lr*100:.2f}%", f"{w_rf*100:.2f}%", f"{w_gbdt*100:.2f}%"]
            })
            st.table(weight_table)
            
        with col_m2:
            st.markdown("#### 💡 模型指標與演算法解讀")
            st.markdown(
                f"- **嶺回歸 (Ridge Regression)**：特徵工程加入 `StandardScaler` 標準化，以確保 L2 正則化懲罰能公平對待氣壓與月份等不同尺度的特徵，抑制共線性引起的權重發散。\n"
                f"- **隨機森林 (Random Forest)**：透過袋裝法 (Bagging) 隨機取樣特徵，對異常天氣干擾有極佳的魯棒性，能輸出特徵重要性權重。\n"
                f"- **梯度提升樹 (HistGradientBoosting)**：擅長發掘複雜且非線性的天氣規律，對於高度偏態的降雨量有強大的特徵表徵能力。\n"
                f"- **堆疊集成模型 (Stacking Ensemble)**：本系統不採用人工硬編碼比例，而是運用 Stacking 技術在獨立時間序列驗證集上，以約束非負的線性迴歸演算法**動態自動學習最優融合權重**。如左表所示，不同的氣候指標會依據自身特性被分派最適合的融合百分比，在報告中更具學術說服力！"
            )
            
        st.markdown("---")
        st.subheader(f"📈 {selected_analysis_cn} AI 預估效能與診斷圖表 (任選指標動態繪圖)")
        
        # 根據側邊欄的 start_date 和 end_date 篩選測試集數據與預測值
        mask_test = (df_test["Date"] >= pd.to_datetime(start_date)) & (df_test["Date"] <= pd.to_datetime(end_date))
        filtered_df_test = df_test[mask_test]
        
        if len(filtered_df_test) == 0:
            test_start_date_str = df_test["Date"].min().strftime('%Y/%m/%d')
            test_end_date_str = df_test["Date"].max().strftime('%Y/%m/%d')
            st.warning(
                f"⚠️ **目前探索的時間區段未包含機器學習模型之測試集數據**\n\n"
                f"本專案之機器學習模型使用前 80% 的歷史數據進行訓練，後 20% 作為測試集進行泛化評估。\n"
                f"當前測試集的時間區段為 **{test_start_date_str} ~ {test_end_date_str}**。\n"
                f"請於左側控制面板將「歷史資料探索時間區段」之時間範圍往後調整（需包含上述區間），以呈現對應之模型成效圖表。"
            )
        else:
            # 取得篩選後的繪圖數據
            y_test = filtered_df_test[analysis_var]
            y_pred_all = test_predictions[analysis_var]["ensemble"]
            # 使用與 df_test 對齊的布林遮罩進行 NumPy 陣列篩選
            y_pred = y_pred_all[mask_test.values]
            dates = filtered_df_test["Date"]
            
            # 1. 特徵重要性
            rf_model = rf_models[analysis_var]
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            # 特徵中文轉換
            feature_names_cn = []
            for name in feature_names:
                if "_lag1" in name:
                    base_col = name.replace("_lag1", "")
                    feature_names_cn.append(f"昨日 {COLUMN_CN_MAP.get(base_col, base_col)}")
                elif "_lag2" in name:
                    base_col = name.replace("_lag2", "")
                    feature_names_cn.append(f"前日 {COLUMN_CN_MAP.get(base_col, base_col)}")
                elif "_diff1" in name:
                    base_col = name.replace("_diff1", "")
                    feature_names_cn.append(f"兩日差 {COLUMN_CN_MAP.get(base_col, base_col)}")
                elif "Month" in name:
                    feature_names_cn.append(f"月份週期因子 ({name.split('_')[-1]})")
                elif "DayOfYear" in name:
                    feature_names_cn.append(f"日曆週期因子 ({name.split('_')[-1]})")
                else:
                    feature_names_cn.append(COLUMN_CN_MAP.get(name, name))
            
            fig_fi, ax_fi = plt.subplots(figsize=(7.5, 5))
            ax_fi.barh([feature_names_cn[i] for i in indices][::-1], importances[indices][::-1], color="#1f77b4")
            ax_fi.set_title(f"{selected_analysis_cn} - 前 10 大預測特徵權重分析", fontsize=11, pad=10)
            ax_fi.set_xlabel("相對重要性權重", fontsize=9)
            ax_fi.tick_params(axis='both', labelsize=9)
            ax_fi.grid(axis='x', linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            # 2. 擬合散佈圖
            fig_sf, ax_sf = plt.subplots(figsize=(7, 5))
            ax_sf.scatter(y_test.values, y_pred, alpha=0.6, color="#2ca02c")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax_sf.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="對角對齊線")
            ax_sf.set_title(f"{selected_analysis_cn} - 預估值與實際值擬合散佈圖", fontsize=11, pad=10)
            ax_sf.set_xlabel("實際值", fontsize=9)
            ax_sf.set_ylabel("預估值", fontsize=9)
            ax_sf.tick_params(axis='both', labelsize=9)
            ax_sf.grid(True, linestyle=":", alpha=0.6)
            ax_sf.legend(fontsize=9)
            plt.tight_layout()
            
            # 3. 實際與預測對比折線圖 (展示篩選區間內最新 90 天)
            sample_len = min(len(filtered_df_test), 90)
            sample_dates = dates.head(sample_len)
            sample_actual = y_test.head(sample_len)
            sample_pred = y_pred[:sample_len]
            
            fig_pc, ax_pc = plt.subplots(figsize=(14, 5))
            ax_pc.plot(sample_dates.values, sample_actual.values, label="實際觀測值", color="#1f77b4", marker='o', alpha=0.8)
            ax_pc.plot(sample_dates.values, sample_pred, label="加權集成預估", color="#ff7f0e", linestyle="--", marker='x', alpha=0.8)
            ax_pc.set_title(f"{selected_analysis_cn} - 實際與預估對比圖 (展示篩選時段內前 {sample_len} 天)", fontsize=11, pad=10)
            ax_pc.set_xlabel("日期", fontsize=9)
            ax_pc.set_ylabel(selected_analysis_cn, fontsize=9)
            ax_pc.tick_params(axis='both', labelsize=9)
            ax_pc.grid(True, linestyle=":", alpha=0.6)
            ax_pc.legend(fontsize=9)
            plt.tight_layout()
            
            # 4. 殘差時序圖
            residuals = y_pred - y_test.values
            sample_residuals = residuals[:sample_len]
            fig_res, ax_res = plt.subplots(figsize=(7, 5))
            ax_res.scatter(sample_dates.values, sample_residuals, color="#d62728", alpha=0.7, edgecolors='grey')
            ax_res.axhline(0, color='black', linestyle='--', lw=1.5)
            ax_res.set_title(f"{selected_analysis_cn} - 預估殘差時序圖 (展示篩選時段內前 {sample_len} 天)", fontsize=11, pad=10)
            ax_res.set_xlabel("日期", fontsize=9)
            ax_res.set_ylabel("預估誤差 (預估 - 實際)", fontsize=9)
            ax_res.tick_params(axis='both', labelsize=9)
            ax_res.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            
            # 5. 誤差分佈直方圖
            fig_err, ax_err = plt.subplots(figsize=(7, 5))
            clean_residuals = residuals[~np.isnan(residuals)]
            if len(clean_residuals) > 0:
                ax_err.hist(clean_residuals, bins=20, density=True, alpha=0.6, color="#9467bd", edgecolor='grey', label="誤差分佈")
                mu_err, std_err = clean_residuals.mean(), clean_residuals.std()
                if std_err > 0:
                    x_err = np.linspace(clean_residuals.min(), clean_residuals.max(), 100)
                    p_err = (1 / (np.sqrt(2 * np.pi) * std_err)) * np.exp(-0.5 * ((x_err - mu_err) / std_err) ** 2)
                    ax_err.plot(x_err, p_err, color="#d62728", linewidth=2.5, label="常態分佈擬合線")
                ax_err.set_title(f"{selected_analysis_cn} - 預估誤差常態分佈檢驗", fontsize=11, pad=10)
                ax_err.set_xlabel("預估誤差 (預估 - 實際)", fontsize=9)
                ax_err.set_ylabel("密度", fontsize=9)
                ax_err.tick_params(axis='both', labelsize=9)
                ax_err.grid(True, linestyle=":", alpha=0.6)
                ax_err.legend(fontsize=9)
            else:
                ax_err.text(0.5, 0.5, "無誤差數據", ha='center', va='center')
            plt.tight_layout()
            
            # 網頁佈局渲染
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("#### A. 特徵重要性分析 (Feature Importance)")
                st.pyplot(fig_fi)
                st.markdown("**說明**：顯示昨日各天氣指標對今日該觀測變數之決定性影響權重。")
            with col_p2:
                st.markdown("#### B. 預估與實際擬合散佈圖 (Scatter Fit)")
                st.pyplot(fig_sf)
                st.markdown("**說明**：散佈點若越集中在紅色對角線上，代表模型預測精度越接近理想預估。")
                
            st.markdown("---")
            st.markdown("#### C. 實際值與預估值折線對比 (Prediction Timeline)")
            st.pyplot(fig_pc)
            st.markdown("**說明**：實線代表中央氣象署的實際日觀測數據，虛線代表集成模型的預估值，兩條線越接近且走勢一致代表擬合成效越佳。")
            
            st.markdown("---")
            col_p3, col_p4 = st.columns(2)
            with col_p3:
                st.markdown("#### D. 預估殘差時序圖 (Residual Plot)")
                st.pyplot(fig_res)
                st.markdown("**說明**：點代表每日誤差（預估 - 實際）。好的模型誤差點應隨機散佈在 0 線上下，無明顯季節規律。")
            with col_p4:
                st.markdown("#### E. 預估誤差分佈直方圖 (Error Distribution)")
                st.pyplot(fig_err)
                st.markdown("**說明**：直方圖展示誤差次數，紅色線為常態分佈擬合。誤差越接近常態分佈，在學術上代表模型剩餘訊息已接近隨機白噪音。")
                
            # 關閉圖表以釋放記憶體
            plt.close(fig_fi)
            plt.close(fig_sf)
            plt.close(fig_pc)
            plt.close(fig_res)
            plt.close(fig_err)
