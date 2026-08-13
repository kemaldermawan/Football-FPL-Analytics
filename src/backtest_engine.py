import pandas as pd
import numpy as np

def run_backtest_evaluation(df, predicted_col="ep_next", actual_col="total_points"):
    """
    Mengevaluasi akurasi metrik proyeksi poin terhadap hasil aktual historis.
    """
    eval_df = df.dropna(subset=[predicted_col, actual_col]).copy()
    
    eval_df[predicted_col] = pd.to_numeric(eval_df[predicted_col], errors="coerce").fillna(0)
    eval_df[actual_col] = pd.to_numeric(eval_df[actual_col], errors="coerce").fillna(0)
    
    y_hat = eval_df[predicted_col].values
    y = eval_df[actual_col].values
    
    if len(y) == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "Correlation": 0.0, "Evaluated_Players": 0}
        
    errors = y_hat - y
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    if np.std(y_hat) > 0 and np.std(y) > 0:
        corr = np.corrcoef(y_hat, y)[0, 1]
    else:
        corr = 0.0
        
    eval_df["Error_Delta"] = eval_df[predicted_col] - eval_df[actual_col]
    
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "Correlation": round(float(corr), 3),
        "Evaluated_Players": int(len(y)),
        "Detailed_DF": eval_df
    }