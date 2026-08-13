import pandas as pd
import numpy as np

def calculate_expected_bps(df):
    """
    Mengalkulasi proyeksi Bonus Points (xBPS) berdasarkan metrik ekspektasi dasar.
    """
    eval_df = df.copy()
    
    # Injeksi kerangka Series aman untuk mencegah tipe data integer memicu galat
    target_cols = ["expected_goals", "expected_assists", "minutes"]
    for col in target_cols:
        if col not in eval_df.columns:
            eval_df[col] = 0.0
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce").fillna(0.0)
        
    def compute_xbps(row):
        xbps = 0.0
        pos = row.get("Position", "MID")
        
        # Injeksi poin fundamental BPS
        xbps += (row["expected_goals"] * 24.0)
        xbps += (row["expected_assists"] * 9.0)
        
        # Estimasi menit bermain menghasilkan rata-rata baseline Pass/Tackle BPS
        if row["minutes"] > 60:
            xbps += 6.0 
            
        # Penyesuaian BPS spesifik posisi (Nirbobol / Penyelamatan)
        if pos in ["DEF", "GKP"]:
            # Asumsi probabilitas nirbobol dasar 25 persen untuk kalkulasi statis
            xbps += (0.25 * 12.0)
            if pos == "GKP":
                xbps += 4.5 # Rata-rata 3 penyelamatan per laga
        elif pos == "FWD":
            xbps += 3.0 # Baseline tembakan tepat sasaran
            
        return round(xbps, 2)

    eval_df["Proj_xBPS"] = eval_df.apply(compute_xbps, axis=1)
    
    # Konversi xBPS menjadi probabilitas Bonus Poin FPL (1-3)
    def map_bps_to_bonus(xbps):
        if xbps > 28.0: return 3.0
        elif xbps > 24.0: return 2.0
        elif xbps > 20.0: return 1.0
        return 0.0
        
    eval_df["Proj_Bonus_Points"] = eval_df["Proj_xBPS"].apply(map_bps_to_bonus)
    
    return eval_df

def optimize_bench_order(squad_df, target_metric="ep_next"):
    """
    Mengurutkan prioritas pemain cadangan berdasarkan ekspektasi poin dan probabilitas bermain.
    """
    bench_df = squad_df.copy()
    
    # Penanganan tipe data yang aman untuk mencegah galat saat kolom absen
    if "chance_of_playing_next_round" not in bench_df.columns:
        bench_df["chance_of_playing_next_round"] = 100.0
        
    if target_metric not in bench_df.columns:
        bench_df[target_metric] = 0.0
        
    # Konversi metrik probabilitas bermain (0-100%)
    bench_df["chance_of_playing"] = pd.to_numeric(bench_df["chance_of_playing_next_round"], errors="coerce").fillna(100.0)
    bench_df[target_metric] = pd.to_numeric(bench_df[target_metric], errors="coerce").fillna(0.0)
    
    # Nilai utilitas adalah ekspektasi poin dikalikan probabilitas tampil riil
    bench_df["Utility_Score"] = bench_df[target_metric] * (bench_df["chance_of_playing"] / 100.0)
    
    # Pemisahan Kiper (Bench 1) dan Outfield (Bench 2-4)
    gkp_bench = bench_df[bench_df["Position"] == "GKP"].sort_values("Utility_Score", ascending=False)
    outfield_bench = bench_df[bench_df["Position"] != "GKP"].sort_values("Utility_Score", ascending=False)
    
    return gkp_bench, outfield_bench