# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_cwa_csv(filepath):
    """
    解析氣象署每日觀測資料 CSV 檔。
    跳過第一列中文標頭，使用第二列英文標頭。
    """
    df_temp = pd.read_csv(filepath, skiprows=[0])
    
    # 從檔名解析測站 ID、年份與月份 (檔名格式: 467110-2026-06.csv)
    filename = os.path.basename(filepath)
    parts = filename.replace(".csv", "").split("-")
    if len(parts) >= 3:
        station_id = parts[0]
        year = int(parts[1])
        month = int(parts[2])
    else:
        station_id = "467110"
        year = 2026
        month = 6
        
    # 過濾非數值的天數列 (例如未發生的天數標記為 -- 或空值)
    df_temp = df_temp[pd.to_numeric(df_temp["ObsTime"], errors="coerce").notna()]
    df_temp["ObsTime"] = df_temp["ObsTime"].astype(int)
    
    # 清理資料型態
    for col in df_temp.columns:
        if col != "ObsTime":
            df_temp[col] = df_temp[col].astype(str).str.strip()
            df_temp[col] = df_temp[col].replace("--", np.nan)
            if col == "Precp":
                df_temp[col] = df_temp[col].replace("T", "0.0")
            # 只有在非時間/字元欄位時，才轉為數值
            if "Time" not in col and "LST" not in col:
                df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce")
                
    # 移除溫度為空的列 (例如本月尚未發生的未來日期)
    df_temp = df_temp.dropna(subset=["Temperature"])
    
    # 建立標準日期物件
    dates = []
    for day in df_temp["ObsTime"]:
        dates.append(pd.to_datetime(f"{year}-{month:02d}-{int(day):02d}"))
    df_temp["Date"] = dates
    df_temp["StationID"] = station_id
    
    # 重新排列欄位，使 Date 與 StationID 位於最前
    cols = ["Date", "StationID"] + [c for c in df_temp.columns if c not in ["Date", "StationID"]]
    return df_temp[cols]

def load_real_cwa_data(folder_path):
    """
    掃描特定目錄下所有的氣象署觀測 CSV 檔案，進行解析與合併。
    """
    if not os.path.exists(folder_path):
        print(f"[*] 找不到真實氣象資料夾: {folder_path}")
        return None
        
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"[*] 真實氣象資料夾 {folder_path} 內無 CSV 檔案。")
        return None
        
    df_list = []
    for f in csv_files:
        try:
            df_single = parse_cwa_csv(f)
            if len(df_single) > 0:
                df_list.append(df_single)
                print(f"[+] 成功讀取並解析氣象觀測檔: {os.path.basename(f)} (共 {len(df_single)} 筆日紀錄)")
        except Exception as e:
            print(f"[-] 解析真實氣象檔案時出錯 {f}: {e}")
            
    if not df_list:
        return None
        
    return pd.concat(df_list, ignore_index=True)

def generate_kinmen_weather_data(filepath=None, force_regenerate=False):
    """
    加載真實的金門氣象署 CSV 資料。完全無任何模擬數據。
    """
    if filepath is None:
        filepath = os.path.join(BASE_DIR, "kinmen_weather_10y.csv")
        
    if os.path.exists(filepath) and not force_regenerate:
        print(f"[*] 偵測到已存在的資料集: {filepath}，直接載入。")
        return pd.read_csv(filepath, parse_dates=['Date'])

    cwa_folder = os.path.join(BASE_DIR, "日月CSV")
    df = load_real_cwa_data(cwa_folder)
    
    if df is None or len(df) == 0:
        raise ValueError("[-] 找不到任何氣象署 CSV 檔案！請放檔案至「金門天氣預測專案/日月CSV」資料夾。")
        
    df = df.sort_values(by=["Date", "StationID"]).reset_index(drop=True)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[+] 真實資料集加載成功！共 {len(df)} 筆日觀測紀錄，已儲存至: {filepath}")
    return df

if __name__ == "__main__":
    generate_kinmen_weather_data(force_regenerate=True)
