"""
q8_classificacao_e_modelos.py — Recomendação de modelo por família (Q8)
=========================================================================

Para cada uma das 5 famílias (F1–F5):
    1. Classifica perfil de demanda via Syntetos-Boylan (ADI × CV²)
       e via CV² isolado (fallback quando não há zeros na série).
    2. Roda benchmark sistemático dos 7 modelos do Anexo F:
       Naive, MA(4), SES, Holt, Winters, ETS, ARIMA.
    3. Recomenda o melhor modelo considerando:
       - Menor RMSE no teste (acurácia)
       - Critério de parcimônia (empate < 5% → simpler wins)
       - Viabilidade operacional (Excel / Python / Cientista dedicado)

Metodologia:
    - Série: Anexo D1 (demanda agregada semanal), 104 semanas por família.
    - Split: 80 semanas treino (1–80), 24 semanas teste (81–104).
    - Métricas: MAE, RMSE, MAPE.

Saídas:
    - outputs/tables/q8_anexo_f_preenchido.csv   (MAE/RMSE/MAPE × modelo × família)
    - outputs/tables/q8_classificacao_familias.csv
    - outputs/tables/q8_matriz_recomendacao.csv
    - outputs/figures/q8_perfil_demanda.png       (scatter ADI × CV²)
    - outputs/figures/q8_comparativo_modelos.png  (heatmap RMSE normalizado)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, Holt, ExponentialSmoothing,
)
from statsmodels.tsa.arima.model import ARIMA

from utils import (
    load_demand_family,
    load_family_profile,
    compute_metrics,
    classify_syntetos_boylan,
    PATH_FIGURES,
    PATH_TABLES,
    CORES,
    MPL_STYLE,
)

plt.rcParams.update(MPL_STYLE)

N_TREINO = 80
N_TESTE = 24
FAMILIAS = ["F1", "F2", "F3", "F4", "F5"]

# ────────────────────────────────────────────────────────────────────────
# Funções de previsão (assinatura uniforme: train, h → pred)
# ────────────────────────────────────────────────────────────────────────
def fc_naive(train, h):
    return np.full(h, train.iloc[-1], dtype=float)


def fc_ma(train, h, janela=4):
    """Média móvel de 4 semanas (repete última média ao longo do horizonte)."""
    ultima_ma = train.iloc[-janela:].mean()
    return np.full(h, ultima_ma, dtype=float)


def fc_ses(train, h):
    m = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(m.forecast(h))


def fc_holt(train, h):
    m = Holt(train, initialization_method="estimated").fit(optimized=True)
    return np.asarray(m.forecast(h))


def fc_winters(train, h, seasonal_periods=52):
    """Holt-Winters aditivo com init='known' (priors neutros)."""
    m = ExponentialSmoothing(
        train, trend="add", seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="known",
        initial_level=float(train.mean()),
        initial_trend=0.0,
        initial_seasonal=[0.0] * seasonal_periods,
    ).fit(optimized=True)
    return np.asarray(m.forecast(h))


def fc_ets(train, h):
    """ETS automático: tenta (A,A,N), (A,A,A) s=13 — retorna o de menor AIC."""
    candidatos = []
    # (A, A, N) — tendência, sem sazonalidade
    try:
        m = ExponentialSmoothing(
            train, trend="add", seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        candidatos.append((m.aic, m))
    except Exception:
        pass
    # (A, A, A) — tendência + sazonalidade trimestral (13 sem) para permitir MLE
    try:
        m = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=13,
            initialization_method="estimated",
        ).fit(optimized=True)
        candidatos.append((m.aic, m))
    except Exception:
        pass
    if not candidatos:
        return fc_holt(train, h)
    melhor = min(candidatos, key=lambda x: x[0])[1]
    return np.asarray(melhor.forecast(h))


def fc_arima(train, h):
    """ARIMA(1,1,1) — modelo genérico simples, compromisso tratável."""
    try:
        m = ARIMA(train, order=(1, 1, 1)).fit()
        return np.asarray(m.forecast(h))
    except Exception:
        return fc_naive(train, h)


MODELOS = {
    "Naive":   fc_naive,
    "MA(4)":   fc_ma,
    "SES":     fc_ses,
    "Holt":    fc_holt,
    "Winters": fc_winters,
    "ETS":     fc_ets,
    "ARIMA":   fc_arima,
}


# ────────────────────────────────────────────────────────────────────────
# Classificação do perfil de demanda
# ────────────────────────────────────────────────────────────────────────
def classify_family(serie: pd.Series, cv_cortes=(0.15, 0.35)) -> dict:
    """
    Combina Syntetos-Boylan (ADI × CV²) com uma classificação adicional
    baseada em CV² quando não há zeros (Vila Aurora: F1/F3/F4 têm ADI=1).

    cv_cortes: (baixo, alto) — separa smooth / variável / erratic.
    """
    sb = classify_syntetos_boylan(serie)
    cv2 = sb["CV2"]

    # Classificação prática (fallback quando ADI ≈ 1)
    if cv2 < cv_cortes[0]:
        pratica = "Smooth puro"
    elif cv2 < cv_cortes[1]:
        pratica = "Moderadamente variável"
    else:
        pratica = "Erratic/Lumpy"
    sb["classificacao_pratica"] = pratica
    return sb


# ────────────────────────────────────────────────────────────────────────
# Benchmark por família
# ────────────────────────────────────────────────────────────────────────
def benchmark_family(familia_cod: str, df_fam: pd.DataFrame) -> dict:
    """Roda os 7 modelos na série de uma família e retorna métricas."""
    serie = (df_fam[df_fam["familia_cod"] == familia_cod]
             .sort_values("semana")["demanda_total"].reset_index(drop=True))
    assert len(serie) == 104, f"{familia_cod} tem {len(serie)} semanas (esperado 104)"

    train = serie.iloc[:N_TREINO]
    teste = serie.iloc[N_TREINO:]
    h = len(teste)

    previsoes, metricas = {}, {}
    for nome, funcao in MODELOS.items():
        try:
            pred = funcao(train, h)
            previsoes[nome] = pred
            metricas[nome] = compute_metrics(teste.values, pred)
        except Exception as e:
            metricas[nome] = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
            previsoes[nome] = np.full(h, np.nan)
            print(f"      ! {familia_cod}-{nome}: erro ({e})")

    perfil = classify_family(serie)

    return {
        "familia": familia_cod,
        "serie": serie,
        "train": train,
        "teste": teste,
        "previsoes": previsoes,
        "metricas": metricas,
        "perfil": perfil,
    }


# ────────────────────────────────────────────────────────────────────────
# Recomendação por família (combina acurácia + parcimônia + operacionalização)
# ────────────────────────────────────────────────────────────────────────
ORDEM_COMPLEXIDADE = ["Naive", "MA(4)", "SES", "Holt", "Winters", "ETS", "ARIMA"]
VIABILIDADE = {
    "Naive":   "Excel",
    "MA(4)":   "Excel",
    "SES":     "Excel (com add-in) / Python",
    "Holt":    "Excel (add-in) / Python",
    "Winters": "Python/R",
    "ETS":     "Python/R",
    "ARIMA":   "Python/R + analista experiente",
}


def recomendar_modelo(metricas: dict, tolerancia: float = 0.05) -> str:
    """Escolhe o modelo mais simples que chega a <5% do menor RMSE."""
    validos = {k: v["RMSE"] for k, v in metricas.items() if not np.isnan(v["RMSE"])}
    if not validos:
        return "Naive"
    melhor_rmse = min(validos.values())
    for nome in ORDEM_COMPLEXIDADE:
        if nome in validos and validos[nome] <= melhor_rmse * (1 + tolerancia):
            return nome
    return min(validos, key=validos.get)


# ────────────────────────────────────────────────────────────────────────
# Visualizações
# ────────────────────────────────────────────────────────────────────────
def plot_perfil_demanda(resultados: list, perfis_teoricos: pd.DataFrame, path_fig):
    """Scatter ADI × CV² com quadrantes Syntetos-Boylan anotados."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Zonas de classificação
    ax.axvline(1.32, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0.49, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(1.05, 0.05, "Smooth", fontsize=11, color="#2E7D32", fontweight="bold", alpha=0.6)
    ax.text(1.5, 0.05, "Intermittent", fontsize=11, color="#F57C00", fontweight="bold", alpha=0.6)
    ax.text(1.05, 0.9, "Erratic", fontsize=11, color="#1976D2", fontweight="bold", alpha=0.6)
    ax.text(1.5, 0.9, "Lumpy", fontsize=11, color="#C62828", fontweight="bold", alpha=0.6)

    for res in resultados:
        fam = res["familia"]
        perf = res["perfil"]
        ax.scatter(perf["ADI"], perf["CV2"], s=250, color=CORES[fam],
                   edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(fam, (perf["ADI"], perf["CV2"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("ADI — Average Demand Interval", fontsize=11)
    ax.set_ylabel("CV² — da demanda não-zero", fontsize=11)
    ax.set_title("Q8 — Perfil de demanda das 5 famílias (Syntetos-Boylan)\n"
                 "Série agregada D1, 104 semanas", fontsize=12, fontweight="bold")
    ax.set_xlim(0.95, 1.7)
    ax.set_ylim(-0.05, 1.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close()


def plot_heatmap_modelos(resultados: list, path_fig):
    """Heatmap de RMSE normalizado (% do melhor modelo por família)."""
    familias = [r["familia"] for r in resultados]
    matriz = np.zeros((len(familias), len(ORDEM_COMPLEXIDADE)))

    for i, r in enumerate(resultados):
        rmses = [r["metricas"][m]["RMSE"] for m in ORDEM_COMPLEXIDADE]
        melhor = np.nanmin(rmses)
        for j, rmse in enumerate(rmses):
            matriz[i, j] = 100 * (rmse - melhor) / melhor if not np.isnan(rmse) else np.nan

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(matriz, cmap="RdYlGn_r", vmin=0, vmax=40, aspect="auto")
    ax.set_xticks(range(len(ORDEM_COMPLEXIDADE)))
    ax.set_xticklabels(ORDEM_COMPLEXIDADE)
    ax.set_yticks(range(len(familias)))
    ax.set_yticklabels(familias)

    for i in range(len(familias)):
        for j in range(len(ORDEM_COMPLEXIDADE)):
            val = matriz[i, j]
            if not np.isnan(val):
                cor = "white" if val > 25 else "black"
                texto = "★" if val == 0 else f"+{val:.0f}%"
                ax.text(j, i, texto, ha="center", va="center",
                        color=cor, fontsize=10,
                        fontweight="bold" if val == 0 else "normal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RMSE acima do melhor modelo (%)", fontsize=10)
    ax.set_title("Q8 — Comparativo de modelos por família\n"
                 "★ = melhor RMSE  |  valor = % acima do melhor",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Q8 — Classificação de Demanda e Recomendação de Modelo por Família")
    print("=" * 70)

    # 1. Carregar dados
    df_fam = load_demand_family()
    df_perfil = load_family_profile()
    print(f"\n[1/5] Anexo D1 carregado: {df_fam.shape[0]} linhas")
    print(f"       Anexo A1 carregado: {len(df_perfil)} famílias")

    # 2. Rodar benchmark nas 5 famílias
    print(f"\n[2/5] Rodando {len(MODELOS)} modelos × {len(FAMILIAS)} famílias...")
    resultados = []
    for fam in FAMILIAS:
        print(f"      • {fam}...")
        res = benchmark_family(fam, df_fam)
        resultados.append(res)

    # 3. Tabelas de saída
    print(f"\n[3/5] Montando tabelas...")

    # 3a. Anexo F preenchido
    linhas_f = []
    for r in resultados:
        for modelo, met in r["metricas"].items():
            linhas_f.append({
                "familia": r["familia"],
                "modelo": modelo,
                "MAE": round(met["MAE"], 2),
                "RMSE": round(met["RMSE"], 2),
                "MAPE_%": round(met["MAPE"], 2),
            })
    df_anexoF = pd.DataFrame(linhas_f)
    df_anexoF.to_csv(PATH_TABLES / "q8_anexo_f_preenchido.csv",
                     index=False, encoding="utf-8-sig")

    # 3b. Classificação de demanda
    linhas_class = []
    for r in resultados:
        perf_teorico = df_perfil[df_perfil["familia_cod"] == r["familia"]].iloc[0]
        linhas_class.append({
            "familia": r["familia"],
            "nome": perf_teorico["familia"],
            "perfil_caso": perf_teorico["perfil"],
            "CV_estrutural_%": perf_teorico["CV%"],
            "CV_observado_%": round(
                100 * r["serie"].std() / r["serie"].mean(), 1),
            "ADI_observado": round(r["perfil"]["ADI"], 3),
            "CV2_observado": round(r["perfil"]["CV2"], 3),
            "n_zeros": r["perfil"]["n_zeros"],
            "classificacao_SB": r["perfil"]["classificacao"],
            "classificacao_pratica": r["perfil"]["classificacao_pratica"],
        })
    df_class = pd.DataFrame(linhas_class)
    df_class.to_csv(PATH_TABLES / "q8_classificacao_familias.csv",
                    index=False, encoding="utf-8-sig")

    # 3c. Matriz de recomendação
    linhas_rec = []
    for r in resultados:
        melhor_modelo = recomendar_modelo(r["metricas"], tolerancia=0.05)
        rmse_melhor = r["metricas"][melhor_modelo]["RMSE"]
        mape_melhor = r["metricas"][melhor_modelo]["MAPE"]
        perf_teorico = df_perfil[df_perfil["familia_cod"] == r["familia"]].iloc[0]
        linhas_rec.append({
            "familia": r["familia"],
            "nome": perf_teorico["familia"],
            "perfil_pratico": r["perfil"]["classificacao_pratica"],
            "modelo_recomendado": melhor_modelo,
            "RMSE_teste": round(rmse_melhor, 1),
            "MAPE_teste_%": round(mape_melhor, 1),
            "viabilidade": VIABILIDADE[melhor_modelo],
            "freq_recomendada": "semanal" if r["familia"] in ["F1", "F4"] else "quinzenal",
        })
    df_rec = pd.DataFrame(linhas_rec)
    df_rec.to_csv(PATH_TABLES / "q8_matriz_recomendacao.csv",
                  index=False, encoding="utf-8-sig")

    # 4. Figuras
    print(f"\n[4/5] Gerando figuras...")
    plot_perfil_demanda(resultados, df_perfil, PATH_FIGURES / "q8_perfil_demanda.png")
    plot_heatmap_modelos(resultados, PATH_FIGURES / "q8_comparativo_modelos.png")

    # 5. Imprimir resumo
    print(f"\n[5/5] Resumo:")
    print("\n• Classificação de demanda:")
    print(df_class[["familia", "CV_observado_%", "ADI_observado", "CV2_observado",
                    "classificacao_SB", "classificacao_pratica"]].to_string(index=False))

    print("\n• Anexo F preenchido (RMSE):")
    pivot = df_anexoF.pivot(index="familia", columns="modelo", values="RMSE")
    pivot = pivot[ORDEM_COMPLEXIDADE]  # ordena colunas
    print(pivot.round(1).to_string())

    print("\n• Matriz de recomendação:")
    print(df_rec[["familia", "perfil_pratico", "modelo_recomendado",
                  "RMSE_teste", "MAPE_teste_%", "viabilidade"]].to_string(index=False))

    print(f"\n✅ Saídas geradas em outputs/tables/ e outputs/figures/")


if __name__ == "__main__":
    main()
