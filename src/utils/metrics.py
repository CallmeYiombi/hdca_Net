import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    pcc, _ = pearsonr(y_true, y_pred)
    scc, _ = spearmanr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "pcc": float(pcc), "spearman": float(scc)}

def compute_loto_summary(drug_metrics: list) -> dict:
    """drug_metrics: list of dicts with 'pcc' key."""
    pccs = np.array([d["pcc"] for d in drug_metrics])
    return {
        "mean_pcc":          float(np.mean(pccs)),
        "median_pcc":        float(np.median(pccs)),
        "iqr_pcc":           float(np.percentile(pccs, 75) - np.percentile(pccs, 25)),
        "n_negative_pcc":    int((pccs < 0).sum()),
        "negative_pcc_ratio": float((pccs < 0).mean()),
        "n_drugs":           len(pccs),
    }
