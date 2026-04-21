"""
q4_benchmarking_sku.py — Benchmarking inicial Q4 (Vila Aurora)
================================================================

Compara métodos Naive vs ETS (SES, Holt, Holt-Winters) em dois SKUs
contrastantes do Anexo D2:
    - F1-001 (Leite Integral 1L)    → smooth, alto giro, shelf life curto
    - F2-003 (Vinho Reserva 750ml)  → erratic, baixo giro, LT longo

Metodologia:
    - Split temporal: 80 semanas treino (1-80), 24 semanas teste (81-104)
    - Modelos: Naive, SES, Holt (aditivo), Holt-Winters aditivo sazonalidade=52
    - Métricas: MAE, RMSE, MAPE (todas no período de teste)
    - Implementação via statsmodels.ExponentialSmoothing

Saídas:
    - outputs/tables/q4_resultados.csv
    - outputs/figures/q4_benchmarking.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

from utils import (
    load_demand_sku,
    compute_metrics,
    PATH_FIGURES,
    PATH_TABLES,
    CORES,
    MPL_STYLE,
)

# ────────────────────────────────────────────────────────────────────────
# Configuração
# ────────────────────────────────────────────────────────────────────────
plt.rcParams.update(MPL_STYLE)

SKUS_ALVO = ["F1-001", "F2-003"]
N_TREINO = 80
N_TESTE = 24
N_TOTAL = N_TREINO + N_TESTE  # 104

PATH_FIG = PATH_FIGURES / "q4_benchmarking.png"
PATH_CSV = PATH_TABLES / "q4_resultados.csv"


# ────────────────────────────────────────────────────────────────────────
# Funções de previsão
# ────────────────────────────────────────────────────────────────────────
def split_train_test(serie: pd.Series, n_train: int = N_TREINO):
    """Split temporal: primeiras n_train observações = treino."""
    return serie.iloc[:n_train], serie.iloc[n_train:]


def forecast_naive(train: pd.Series, h: int) -> np.ndarray:
    """Previsão Naive: repete o último valor observado."""
    return np.full(h, train.iloc[-1], dtype=float)


def forecast_ses(train: pd.Series, h: int) -> np.ndarray:
    """Simple Exponential Smoothing com α otimizado por MLE."""
    model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(model.forecast(h))


def forecast_holt(train: pd.Series, h: int) -> np.ndarray:
    """Holt linear (nível + tendência, α e β otimizados)."""
    model = Holt(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(model.forecast(h))


def forecast_holt_winters(train: pd.Series, h: int, seasonal_periods: int = 52) -> np.ndarray:
    """
    Holt-Winters aditivo (nível + tendência + sazonalidade anual).
    
    Nota: com 80 semanas de treino e período=52, temos apenas ~1,5 ciclo
    sazonal, insuficiente para a heurística padrão. Usamos initialization
    'known' com priors neutros (nível = média, tendência e sazonais = 0)
    e deixamos o otimizador MLE refinar os parâmetros α, β, γ.
    """
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="known",
        initial_level=float(train.mean()),
        initial_trend=0.0,
        initial_seasonal=[0.0] * seasonal_periods,
    ).fit(optimized=True)
    return np.asarray(model.forecast(h))


# ────────────────────────────────────────────────────────────────────────
# Benchmark por SKU
# ────────────────────────────────────────────────────────────────────────
def benchmark_sku(sku: str, df_sku: pd.DataFrame) -> dict:
    """Roda os 4 modelos no SKU e retorna dict com previsões e métricas."""
    serie = df_sku[df_sku["sku"] == sku].sort_values("semana")["demanda_total"].reset_index(drop=True)
    assert len(serie) == N_TOTAL, f"SKU {sku} não tem 104 semanas (tem {len(serie)})"

    train, teste = split_train_test(serie)
    h = len(teste)

    previsoes = {
        "Naive":       forecast_naive(train, h),
        "SES":         forecast_ses(train, h),
        "Holt":        forecast_holt(train, h),
        "Holt-Winters": forecast_holt_winters(train, h),
    }

    metricas = {
        nome: compute_metrics(teste.values, pred)
        for nome, pred in previsoes.items()
    }

    # SKU metadata
    nome_produto = df_sku[df_sku["sku"] == sku]["nome_produto"].iloc[0]
    custo = df_sku[df_sku["sku"] == sku]["custo_R$"].iloc[0]

    return {
        "sku": sku,
        "nome": nome_produto,
        "custo_unit": custo,
        "serie": serie,
        "train": train,
        "teste": teste,
        "previsoes": previsoes,
        "metricas": metricas,
    }


def calc_estoque_seguranca(rmse: float, lt_semanas: float, z: float = 1.645) -> float:
    """ES = z × RMSE × √LT (nível de serviço 95% default)."""
    return z * rmse * np.sqrt(lt_semanas)


# ────────────────────────────────────────────────────────────────────────
# Visualização
# ────────────────────────────────────────────────────────────────────────
def plot_benchmarking(resultados: list, path_fig) -> None:
    """Figura 2×2: séries com previsões + barras de RMSE por SKU."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for i, res in enumerate(resultados):
        # Linha 0: série histórica + previsões no período de teste
        ax = axes[0, i]
        ax.plot(res["serie"].index, res["serie"].values,
                color=CORES["real"], linewidth=1.2, label="Real", alpha=0.8)
        ax.axvline(N_TREINO - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.text(N_TREINO - 2, ax.get_ylim()[1] * 0.95, "treino", ha="right", fontsize=8, color="gray")
        ax.text(N_TREINO + 2, ax.get_ylim()[1] * 0.95, "teste", ha="left", fontsize=8, color="gray")

        idx_teste = res["teste"].index
        cor_map = {"Naive": "naive", "SES": "ses", "Holt": "holt", "Holt-Winters": "hw"}
        for nome, pred in res["previsoes"].items():
            ax.plot(idx_teste, pred, color=CORES[cor_map[nome]],
                    linewidth=1.4, label=nome, alpha=0.85)
        ax.set_title(f"{res['sku']} — {res['nome']}", fontweight="bold")
        ax.set_xlabel("Semana")
        ax.set_ylabel("Demanda (un)")
        ax.legend(fontsize=8, loc="upper left")

        # Linha 1: comparativo de RMSE
        ax = axes[1, i]
        nomes = list(res["metricas"].keys())
        rmses = [res["metricas"][n]["RMSE"] for n in nomes]
        mapes = [res["metricas"][n]["MAPE"] for n in nomes]
        cores_barra = [CORES[cor_map[n]] for n in nomes]
        bars = ax.bar(nomes, rmses, color=cores_barra, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("RMSE (un)")
        ax.set_title(f"Erro no período de teste  |  melhor: "
                     f"{nomes[int(np.argmin(rmses))]}", fontweight="bold")
        # Anotar MAPE acima de cada barra
        for bar, mape in zip(bars, mapes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"MAPE\n{mape:.1f}%", ha="center", fontsize=8)
        ax.margins(y=0.15)

    fig.suptitle("Q4 — Benchmarking Naive vs ETS em SKUs contrastantes",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Q4 — Benchmarking Inicial: Naive vs ETS")
    print("=" * 70)

    # 1. Carregar dados
    df_sku = load_demand_sku()
    print(f"\n[1/4] Dados SKU carregados: {df_sku.shape[0]} linhas × {df_sku.shape[1]} colunas")

    # 2. Rodar benchmark em cada SKU
    resultados = []
    print(f"\n[2/4] Rodando 4 modelos em {len(SKUS_ALVO)} SKUs...")
    for sku in SKUS_ALVO:
        res = benchmark_sku(sku, df_sku)
        resultados.append(res)
        print(f"      ✓ {sku} ({res['nome']})")

    # 3. Consolidar tabela de resultados
    print(f"\n[3/4] Consolidando tabela de resultados...")
    linhas = []
    for res in resultados:
        for modelo, met in res["metricas"].items():
            linhas.append({
                "SKU": res["sku"],
                "Produto": res["nome"],
                "Modelo": modelo,
                "MAE": round(met["MAE"], 2),
                "RMSE": round(met["RMSE"], 2),
                "MAPE_%": round(met["MAPE"], 2),
            })
    df_res = pd.DataFrame(linhas)
    df_res.to_csv(PATH_CSV, index=False, encoding="utf-8-sig")

    print("\n" + df_res.to_string(index=False))

    # 4. Gerar figura
    print(f"\n[4/4] Gerando figura...")
    plot_benchmarking(resultados, PATH_FIG)

    # Identificar melhor modelo por SKU e calcular ES implicado
    print("\n" + "─" * 70)
    print("  Melhor modelo por SKU e estoque de segurança implicado (z=1,645)")
    print("─" * 70)
    lt_mapa = {"F1-001": 2, "F2-003": 14}  # LT em dias do Anexo G, convertendo:
    lt_semanas_mapa = {"F1-001": 2/7, "F2-003": 14/7}  # LT em semanas
    for res in resultados:
        melhor = min(res["metricas"].items(), key=lambda x: x[1]["RMSE"])
        nome_m, met_m = melhor
        lt_sem = lt_semanas_mapa[res["sku"]]
        es = calc_estoque_seguranca(met_m["RMSE"], lt_sem)
        es_rs = es * res["custo_unit"]
        print(f"  {res['sku']} → {nome_m}: RMSE={met_m['RMSE']:.1f}, "
              f"ES(95%) ≈ {es:.0f} un (R$ {es_rs:,.0f})")

    print(f"\n✅ Saídas geradas:")
    print(f"   - {PATH_CSV.relative_to(PATH_CSV.parent.parent.parent)}")
    print(f"   - {PATH_FIG.relative_to(PATH_FIG.parent.parent.parent)}")


if __name__ == "__main__":
    main()
