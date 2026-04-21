"""
Q9 — Cadeia causal: erro de previsão → estoque de segurança → ruptura →
     viagens extras / descarte / retrabalho.

Lógica geral:
    O RMSE do modelo vencedor (Q8) alimenta o cálculo do estoque de segurança
    necessário para atingir o nível de serviço alvo de cada família.
    A diferença entre o estoque de segurança real e o necessário determina
    a probabilidade de ruptura, que por sua vez propaga custos operacionais:
    viagens extras de reabastecimento, descarte por vencimento (perecíveis)
    e retrabalho de pedidos parciais.

Famílias de foco:
    F1 — Perecíveis    (nível de serviço alvo z = 97%, lead time = 1 semana)
    F3 — Mercearia     (nível de serviço alvo z = 95%, lead time = 2 semanas)

Parâmetros de entrada: Anexo G do arquivo Excel principal.

Saídas:
    outputs/tables/q9_cadeia_causal.csv    — tabela de impactos por família
    outputs/figures/q9_cadeia_causal.png   — diagrama visual da cadeia causal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import (
    load_demand_family,
    load_simulation_params,
    compute_metrics,
    PATH_DATA,
)

OUT_TABLES = Path("outputs/tables")
OUT_FIGURES = Path("outputs/figures")


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def compute_safety_stock(rmse: float, lead_time: int, z: float) -> float:
    """
    Calcula o estoque de segurança necessário dado o erro de previsão.

    Fórmula: SS = z × RMSE × √lead_time

    Parameters
    ----------
    rmse : float
        Raiz do erro quadrático médio do modelo de previsão (em unidades).
    lead_time : int
        Tempo de reposição em semanas.
    z : float
        Fator de serviço (ex.: 1.88 para 97%, 1.645 para 95%).

    Returns
    -------
    float
        Estoque de segurança em unidades.
    """
    pass


def simulate_service_level(demand: pd.Series, ss: float) -> dict:
    """
    Simula o nível de serviço empírico dado uma série de demanda e
    um estoque de segurança fixo, contando semanas sem ruptura.

    Parameters
    ----------
    demand : pd.Series
        Série de demanda semanal da família.
    ss : float
        Estoque de segurança disponível (unidades).

    Returns
    -------
    dict
        {'fill_rate': float,  # % de semanas sem ruptura
         'n_stockouts': int,  # número de semanas com ruptura
         'total_weeks': int}
    """
    pass


def propagate_costs(stockouts: int, params: dict) -> dict:
    """
    Propaga o número de rupturas nos custos operacionais via cadeia causal.

    Parameters
    ----------
    stockouts : int
        Número de semanas com ruptura no período simulado.
    params : dict
        Parâmetros da família vindos do Anexo G:
        custo_viagem_extra, custo_descarte_unit, custo_retrabalho_unit, etc.

    Returns
    -------
    dict
        {'custo_viagens_extras': float,
         'custo_descarte': float,
         'custo_retrabalho': float,
         'custo_total': float}
    """
    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Carregar séries de demanda e parâmetros do Anexo G
    df_demand = load_demand_family(PATH_DATA)
    params = load_simulation_params(PATH_DATA)

    familias_foco = {
        "F1": {"z": 1.88, "lead_time": 1},   # perecíveis, 97%
        "F3": {"z": 1.645, "lead_time": 2},  # mercearia, 95%
    }

    resultados = []

    for familia, cfg in familias_foco.items():
        serie = df_demand[familia]

        # 2. Recuperar RMSE do modelo vencedor (salvo pelo q8)
        # TODO: ler de outputs/tables/anexo_f_preenchido.csv

        # 3. Calcular estoque de segurança necessário
        ss = compute_safety_stock(
            rmse=None,  # TODO: substituir pelo RMSE do Q8
            lead_time=cfg["lead_time"],
            z=cfg["z"],
        )

        # 4. Simular nível de serviço empírico
        sl = simulate_service_level(serie, ss)

        # 5. Propagar rupturas nos custos operacionais
        custos = propagate_costs(sl["n_stockouts"], params[familia])

        resultados.append({
            "familia": familia,
            "ss_necessario": ss,
            **sl,
            **custos,
        })

    # 6. Salvar tabela de impactos
    df_resultado = pd.DataFrame(resultados)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(OUT_TABLES / "q9_cadeia_causal.csv", index=False)

    # 7. Gerar e salvar diagrama visual da cadeia causal
    # TODO: gerar figura da cadeia com anotações de valores

    print("Q9 concluído. Saídas salvas em outputs/tables/ e outputs/figures/.")
