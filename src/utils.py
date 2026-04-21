"""
utils.py — Funções auxiliares do projeto Vila Aurora (ENG4550 - PI5)
=====================================================================

Loaders dos Anexos, métricas de previsão e constantes compartilhadas
entre os scripts Q4, Q8 e Q9.

Fontes de dados (Anexos):
    - A1_Familias          → perfil estrutural das 5 famílias
    - D1_Demanda_Familia   → 104 semanas agregadas por família
    - D2_Demanda_SKU       → 104 semanas por SKU (27 SKUs)
    - G_Sim_Params         → custos, lead times e parâmetros de simulação
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# Caminhos (relativos à raiz do repositório — funcionam em qualquer OS)
# ────────────────────────────────────────────────────────────────────────
PATH_REPO = Path(__file__).resolve().parent.parent
PATH_DATA = PATH_REPO / "data" / "raw" / "Vila_Aurora_Grupo_09.xlsx"
PATH_FIGURES = PATH_REPO / "outputs" / "figures"
PATH_TABLES = PATH_REPO / "outputs" / "tables"

# Garante que pastas de saída existam
PATH_FIGURES.mkdir(parents=True, exist_ok=True)
PATH_TABLES.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────
def load_family_profile(path: Path = PATH_DATA) -> pd.DataFrame:
    """Anexo A1 — parâmetros estruturais das 5 famílias."""
    df = pd.read_excel(path, sheet_name="A1_Familias")
    df["familia_cod"] = df["familia"].str[:2]
    return df


def load_demand_family(path: Path = PATH_DATA) -> pd.DataFrame:
    """Anexo D1 — demanda agregada semanal por família (104 semanas × 5)."""
    df = pd.read_excel(path, sheet_name="D1_Demanda_Familia")
    df["familia_cod"] = df["familia"].str[:2]
    return df


def load_demand_sku(path: Path = PATH_DATA) -> pd.DataFrame:
    """Anexo D2 — demanda semanal por SKU."""
    df = pd.read_excel(path, sheet_name="D2_Demanda_SKU")
    return df


def load_simulation_params(path: Path = PATH_DATA) -> pd.DataFrame:
    """Anexo G — parâmetros de simulação (custos, LT, emissões)."""
    df = pd.read_excel(path, sheet_name="G_Sim_Params")
    df["familia_cod"] = df["familia"].str[:2]
    return df


def load_kpis(path: Path = PATH_DATA) -> pd.DataFrame:
    """Anexo E — KPIs históricos (24 meses agregados)."""
    return pd.read_excel(path, sheet_name="E_KPIs")


# ────────────────────────────────────────────────────────────────────────
# Métricas de previsão
# ────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE e MAPE. MAPE ignora divisões por zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    if mask.any():
        mape = float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def classify_syntetos_boylan(serie: pd.Series) -> dict:
    """
    Classifica série temporal via ADI × CV² (Syntetos-Boylan, 2005).
    
    - ADI < 1,32 & CV² < 0,49  → Smooth
    - ADI ≥ 1,32 & CV² < 0,49  → Intermittent
    - ADI < 1,32 & CV² ≥ 0,49  → Erratic
    - ADI ≥ 1,32 & CV² ≥ 0,49  → Lumpy
    """
    vals = np.asarray(serie, dtype=float)
    n = len(vals)
    nz = (vals > 0).sum()
    adi = n / nz if nz > 0 else np.inf
    nonzero = vals[vals > 0]
    if len(nonzero) > 1:
        cv2 = (nonzero.std(ddof=1) / nonzero.mean()) ** 2
    else:
        cv2 = np.inf
    if adi < 1.32 and cv2 < 0.49:
        cls = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        cls = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        cls = "Erratic"
    else:
        cls = "Lumpy"
    return {"ADI": adi, "CV2": cv2, "classificacao": cls, "n_zeros": int(n - nz)}


# ────────────────────────────────────────────────────────────────────────
# Paleta de cores consistente entre figuras
# ────────────────────────────────────────────────────────────────────────
CORES = {
    "real":  "#333333",
    "naive": "#E74C3C",
    "ma":    "#F39C12",
    "ses":   "#3498DB",
    "holt":  "#2ECC71",
    "hw":    "#9B59B6",
    "ets":   "#1ABC9C",
    "arima": "#C0392B",
    "F1": "#2E86AB",
    "F2": "#A23B72",
    "F3": "#F18F01",
    "F4": "#3BA99C",
    "F5": "#C73E1D",
}

# Configuração visual padrão
MPL_STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}
