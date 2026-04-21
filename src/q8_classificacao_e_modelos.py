"""
Q8 — Classificação de perfil de demanda e benchmark de modelos de previsão.

Etapa 1 — Classificação via Syntetos-Boylan (2005):
    Para cada uma das 5 famílias, calcula o Intervalo Médio entre Demandas
    (ADI) e o Coeficiente de Variação ao Quadrado (CV²) e enquadra a série
    em uma das quatro categorias: Suave, Errática, Intermitente ou Agitada.

Etapa 2 — Benchmark de modelos (Anexo F):
    Aplica backtest 70% treino / 30% teste nos seguintes 7 modelos:
        1. Naïve (último valor)
        2. Média Móvel de ordem 4 — MA(4)
        3. Suavização Exponencial Simples — SES (α otimizado)
        4. Holt (tendência linear)
        5. Holt-Winters (tendência + sazonalidade multiplicativa)
        6. ETS automático (statsmodels)
        7. ARIMA (ordem selecionada por AIC)
    Métricas comparadas: MAE, RMSE, MAPE.

Saídas:
    outputs/tables/anexo_f_preenchido.csv  — tabela comparativa de modelos
    outputs/figures/q8_<familia>_forecast.png — gráfico treino/teste/previsão
    outputs/figures/q8_syntetos_boylan.png   — diagrama de classificação ADI×CV²
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from utils import (
    load_demand_family,
    compute_metrics,
    PATH_DATA,
)

OUT_TABLES = Path("outputs/tables")
OUT_FIGURES = Path("outputs/figures")


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def classify_demand_pattern(series: pd.Series) -> dict:
    """
    Classifica o perfil de demanda de uma série temporal segundo o
    diagrama de Syntetos-Boylan: calcula ADI e CV² e retorna a categoria.

    Parameters
    ----------
    series : pd.Series
        Série de demanda semanal de uma família.

    Returns
    -------
    dict
        {'ADI': float, 'CV2': float, 'categoria': str}
        Categorias possíveis: 'Suave', 'Errática', 'Intermitente', 'Agitada'.
    """
    pass


def run_benchmark(series: pd.Series) -> pd.DataFrame:
    """
    Executa o backtest 70/30 dos 7 modelos sobre a série fornecida.

    Parameters
    ----------
    series : pd.Series
        Série de demanda semanal de uma família.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas [modelo, MAE, RMSE, MAPE] — uma linha por modelo.
    """
    pass


def recommend_model(metrics_df: pd.DataFrame) -> str:
    """
    Seleciona o modelo recomendado com base no menor RMSE da tabela de métricas.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Saída de run_benchmark() para uma família.

    Returns
    -------
    str
        Nome do modelo recomendado.
    """
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Carregar séries de demanda por família (Anexo D)
    df_demand = load_demand_family(PATH_DATA)

    familias = ["F1", "F2", "F3", "F4", "F5"]
    resultados = []

    for familia in familias:
        serie = df_demand[familia]

        # 2. Classificar perfil de demanda (Syntetos-Boylan)
        classificacao = classify_demand_pattern(serie)

        # 3. Rodar benchmark dos 7 modelos com backtest 70/30
        metricas = run_benchmark(serie)

        # 4. Identificar modelo recomendado pelo menor RMSE
        modelo_rec = recommend_model(metricas)

        # 5. Salvar gráfico treino/teste/previsão para a família
        # TODO: gerar e salvar figura em OUT_FIGURES

        metricas["familia"] = familia
        metricas["categoria"] = classificacao["categoria"]
        metricas["modelo_recomendado"] = modelo_rec
        resultados.append(metricas)

    # 6. Consolidar e salvar Anexo F preenchido
    anexo_f = pd.concat(resultados, ignore_index=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    anexo_f.to_csv(OUT_TABLES / "anexo_f_preenchido.csv", index=False)

    # 7. Salvar diagrama de classificação ADI × CV²
    # TODO: gerar e salvar figura em OUT_FIGURES

    print("Q8 concluído. Saídas salvas em outputs/tables/ e outputs/figures/.")
