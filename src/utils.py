"""
Utilitários compartilhados entre os scripts do case Vila Aurora Alimentos.

Responsabilidades:
- Leitura das abas relevantes do arquivo Excel principal
- Leitura do CSV de demanda semanal por SKU
- Cálculo das métricas de erro de previsão (MAE, RMSE, MAPE)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Caminho para o arquivo de dados principal
PATH_DATA = Path("data/raw/Vila_Aurora_Grupo_09.xlsx")
PATH_DATA_SKU = Path("data/raw/Vila_Aurora_Grupo_09_D2_Demanda_SKU.csv")


def load_demand_family(path: Path = PATH_DATA) -> pd.DataFrame:
    """
    Carrega a série de demanda agregada por família a partir da aba "Anexo D".

    Parameters
    ----------
    path : Path
        Caminho para o arquivo Excel do case.

    Returns
    -------
    pd.DataFrame
        DataFrame com índice temporal (semanas) e uma coluna por família
        (F1 a F5).
    """
    pass


def load_demand_sku(path: Path = PATH_DATA_SKU) -> pd.DataFrame:
    """
    Carrega a série de demanda semanal por SKU a partir do CSV D2.

    Parameters
    ----------
    path : Path
        Caminho para o arquivo CSV de demanda por SKU.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas [semana, sku_id, demanda].
    """
    pass


def load_family_profile(path: Path = PATH_DATA) -> pd.DataFrame:
    """
    Carrega o perfil de cada família (lead time, custo, lote mínimo etc.)
    a partir da aba "Anexo A".

    Parameters
    ----------
    path : Path
        Caminho para o arquivo Excel do case.

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por família com os atributos do Anexo A.
    """
    pass


def load_simulation_params(path: Path = PATH_DATA) -> dict:
    """
    Carrega os parâmetros de simulação definidos no Anexo G
    (nível de serviço z, lead times, custos unitários etc.).

    Parameters
    ----------
    path : Path
        Caminho para o arquivo Excel do case.

    Returns
    -------
    dict
        Dicionário com chaves correspondentes às famílias (F1–F5) e
        sub-chaves para cada parâmetro (z, lead_time, custo_ruptura, ...).
    """
    pass


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula as métricas de erro entre valores reais e previstos.

    Parameters
    ----------
    y_true : np.ndarray
        Valores reais da série de demanda.
    y_pred : np.ndarray
        Valores previstos pelo modelo.

    Returns
    -------
    dict
        Dicionário com as chaves:
        - 'MAE'  : erro médio absoluto
        - 'RMSE' : raiz do erro quadrático médio
        - 'MAPE' : erro percentual absoluto médio (%)
    """
    pass
