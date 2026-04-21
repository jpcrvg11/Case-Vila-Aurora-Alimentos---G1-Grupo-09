"""
q9_cadeia_causal.py — Da previsão ao impacto operacional (Q9)
================================================================

Demonstra a cadeia causal:

    Erro de previsão (RMSE)
         ↓
    Estoque de segurança requerido (SS = z·RMSE·√LT)
         ↓
    Rupturas quando SS é insuficiente (simulação empírica no teste)
         ↓
    Impacto operacional:
        - Viagens extras emergenciais (R$ 1.250 + 42 kgCO₂ cada)
        - Descarte de perecíveis (para F1, shelf life 7-15 dias)
        - Vendas perdidas (ruptura × preço unitário × margem)
        - Retrabalho (horas-extras no CD)

Famílias analisadas (par contrastante):
    F1 — Perecíveis  (smooth, alto volume, LT curto, shelf life crítico)
    F2 — Premium     (erratic, baixo volume, LT longo, alto custo unitário)

Parâmetros (Anexo G):
    - F1: LT=2 dias, custo R$ 8.50/un, ruptura R$ 10.41/un, descarte R$ 8.50/un
    - F2: LT=14 dias, custo R$ 42.00/un, ruptura R$ 66.15/un, descarte R$ 42.00/un
    - Viagem extra: R$ 1.250 + 42 kgCO₂
    - Custo de pedido: R$ 180

Níveis de serviço alvo (KPI Fill Rate ≥ 97%):
    - F1: 97%  (z = 1,881)
    - F2: 90%  (z = 1,282) — tolera mais risco pelo alto custo de ES

Saídas:
    - outputs/tables/q9_cadeia_causal.csv
    - outputs/tables/q9_impacto_anual.csv
    - outputs/figures/q9_cadeia_causal.png
    - outputs/figures/q9_impacto_financeiro.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

from utils import (
    load_demand_family,
    load_simulation_params,
    load_family_profile,
    compute_metrics,
    PATH_FIGURES,
    PATH_TABLES,
    CORES,
    MPL_STYLE,
)

plt.rcParams.update(MPL_STYLE)

N_TREINO = 80
N_TESTE = 24

# Níveis de serviço alvo e z correspondente
NIVEL_SERVICO = {"F1": 0.97, "F2": 0.90}
Z_SCORE = {f: norm.ppf(p) for f, p in NIVEL_SERVICO.items()}


# ────────────────────────────────────────────────────────────────────────
# Previsão (usa melhor modelo identificado em Q8)
# ────────────────────────────────────────────────────────────────────────
def previsao_melhor_modelo(serie: pd.Series, familia: str) -> tuple:
    """
    Retorna (predicao_teste, rmse_backtest) usando o melhor modelo de Q8:
        F1 → Winters (capta sazonalidade)
        F2 → SES     (série erratic, suavização alta regularização)
    """
    train = serie.iloc[:N_TREINO]
    teste = serie.iloc[N_TREINO:]
    h = len(teste)

    if familia == "F1":
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=52,
            initialization_method="known",
            initial_level=float(train.mean()),
            initial_trend=0.0,
            initial_seasonal=[0.0] * 52,
        ).fit(optimized=True)
    else:  # F2
        model = SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True)

    pred = np.asarray(model.forecast(h))
    metricas = compute_metrics(teste.values, pred)
    return pred, metricas["RMSE"], teste.values


# ────────────────────────────────────────────────────────────────────────
# Cadeia causal: calcula elos
# ────────────────────────────────────────────────────────────────────────
def elo_1_estoque_seguranca(rmse: float, lt_dias: float, z: float) -> float:
    """ES = z × RMSE × √LT (LT convertido para semanas)."""
    lt_semanas = lt_dias / 7
    return z * rmse * np.sqrt(lt_semanas)


def elo_2_simular_rupturas(real: np.ndarray, pred: np.ndarray, ss: float) -> dict:
    """
    Simula operação no período de teste:
        estoque_projetado = pred + ss
        ruptura na semana t se real[t] > estoque_projetado[t]
        un_em_falta[t] = max(0, real[t] - estoque_projetado[t])
    Retorna contagem e volume de rupturas.
    """
    estoque = pred + ss
    faltas = np.maximum(0, real - estoque)
    n_rupturas = int((faltas > 0).sum())
    un_faltantes = float(faltas.sum())
    semanas = len(real)
    return {
        "n_rupturas": n_rupturas,
        "pct_semanas_ruptura": 100 * n_rupturas / semanas,
        "un_faltantes": un_faltantes,
        "fill_rate_sim": 100 * (1 - un_faltantes / real.sum()),
    }


def elo_3_propagar_custos(ruptura: dict, familia: str, params_fam: dict,
                          perf_fam: dict, semanas: int) -> dict:
    """
    Converte rupturas em impacto financeiro e ambiental.

    Pressupostos:
        - Cada semana com ruptura em F1 (perecível) gera 1 viagem emergencial.
        - Em F2 (LT=14d), ruptura não pode ser resolvida com viagem extra (lead time do importador);
          toda a ruptura vira venda perdida.
        - Venda perdida = un × preço_venda, onde preço = custo × (1 + margem).
    """
    preco_venda = perf_fam["custo_R$"] * (1 + perf_fam["margem%"] / 100)

    # Venda perdida (toda ruptura vira venda perdida)
    venda_perdida = ruptura["un_faltantes"] * preco_venda

    # Viagens extras: só F1 (LT curto permite resolver)
    n_viagens = ruptura["n_rupturas"] if familia == "F1" else 0
    custo_viagens = n_viagens * 1250
    co2_viagens = n_viagens * 42

    # Custo de ruptura (penalidade operacional — Anexo G)
    custo_ruptura_op = ruptura["un_faltantes"] * params_fam["p_R$/un"]

    # Escalar para anual (proporcional: 24 semanas → 52 semanas)
    escala_anual = 52 / semanas

    return {
        "venda_perdida_teste_R$": venda_perdida,
        "custo_ruptura_teste_R$": custo_ruptura_op,
        "n_viagens_extras_teste": n_viagens,
        "custo_viagens_teste_R$": custo_viagens,
        "co2_viagens_teste_kg": co2_viagens,
        "venda_perdida_anual_R$": venda_perdida * escala_anual,
        "custo_ruptura_anual_R$": custo_ruptura_op * escala_anual,
        "n_viagens_anual": n_viagens * escala_anual,
        "custo_viagens_anual_R$": custo_viagens * escala_anual,
        "co2_viagens_anual_kg": co2_viagens * escala_anual,
    }


def elo_4_descarte_pereciveis(pred: np.ndarray, real: np.ndarray,
                              params_fam: dict, shelf_life_sem: float) -> dict:
    """
    Estima descarte quando previsão > demanda real e shelf life é curto.
    Para F1 (shelf life 2 semanas), superestimar gera descarte.
    Para F2 (shelf life 30 semanas), superestimar vira estoque, não descarte.
    """
    if shelf_life_sem > 8:  # produtos não-perecíveis
        return {"un_descartadas": 0.0, "custo_descarte_R$": 0.0}
    super_previsao = np.maximum(0, pred - real)
    # Apenas fração do excesso vira descarte de fato (taxa de shelf life)
    # heurística: excesso semanal × shelf_life_ratio
    frac_descarte = min(1.0, 1.0 / shelf_life_sem)
    un_descartadas = super_previsao.sum() * frac_descarte
    custo = un_descartadas * params_fam["desc_R$/un"]
    return {"un_descartadas": un_descartadas, "custo_descarte_R$": custo}


# ────────────────────────────────────────────────────────────────────────
# Visualização 1 — Diagrama de cadeia causal
# ────────────────────────────────────────────────────────────────────────
def plot_cadeia_causal(resultados: list, path_fig):
    """
    Diagrama de fluxo: RMSE → SS → rupturas → impactos,
    comparando F1 e F2 lado a lado.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))

    etapas = ["RMSE\n(erro de previsão)",
              "Estoque de\nSegurança",
              "Rupturas\n(24 sem teste)",
              "Impacto operacional"]
    y_pos = [0.88, 0.68, 0.46, 0.22]

    for ax, res in zip(axes, resultados):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fam = res["familia"]
        cor = CORES[fam]
        nome_completo = {"F1": "F1 — Perecíveis", "F2": "F2 — Premium"}[fam]
        perfil = {"F1": "smooth, LT curto, shelf life crítico",
                  "F2": "erratic, LT longo, alto custo"}[fam]

        ax.text(0.5, 0.975, nome_completo, ha="center", va="top",
                fontsize=14, fontweight="bold", color=cor)
        ax.text(0.5, 0.935, perfil, ha="center", va="top",
                fontsize=9, style="italic", color="#555")

        # Caixas da cadeia
        textos = [
            f"RMSE = {res['rmse']:.0f} un/sem\n"
            f"(modelo: {res['modelo']})",

            f"SS = z · RMSE · √LT\n"
            f"= {res['z']:.2f} × {res['rmse']:.0f} × √{res['lt_sem']:.2f}\n"
            f"= {res['ss']:.0f} un\n"
            f"(NS alvo: {int(NIVEL_SERVICO[fam]*100)}%)",

            f"{res['ruptura']['n_rupturas']} semanas "
            f"({res['ruptura']['pct_semanas_ruptura']:.1f}%)\n"
            f"{res['ruptura']['un_faltantes']:.0f} un em falta\n"
            f"Fill rate simulado: {res['ruptura']['fill_rate_sim']:.1f}%",

            f"Venda perdida: R$ {res['impacto']['venda_perdida_teste_R$']:,.0f}\n"
            f"Viagens extras: {res['impacto']['n_viagens_extras_teste']}\n"
            f"({res['impacto']['custo_viagens_teste_R$']:,.0f} R$ + "
            f"{res['impacto']['co2_viagens_teste_kg']:.0f} kg CO₂)\n"
            f"Descarte estimado: R$ {res['descarte']['custo_descarte_R$']:,.0f}",
        ]

        for y, texto, etapa in zip(y_pos, textos, etapas):
            box = FancyBboxPatch(
                (0.05, y - 0.08), 0.9, 0.14,
                boxstyle="round,pad=0.015",
                facecolor=cor, edgecolor="black",
                alpha=0.18, linewidth=1,
            )
            ax.add_patch(box)
            ax.text(0.05, y + 0.07, etapa,
                    fontsize=9, fontweight="bold",
                    color="#444", style="italic")
            ax.text(0.5, y - 0.01, texto,
                    ha="center", va="center",
                    fontsize=10, family="monospace")

        # Setas
        for i in range(len(y_pos) - 1):
            y_top = y_pos[i] - 0.08
            y_bot = y_pos[i + 1] + 0.06
            ax.annotate("", xy=(0.5, y_bot), xytext=(0.5, y_top),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    fig.suptitle("Q9 — Cadeia causal: erro de previsão → impacto operacional",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# Visualização 2 — Impacto financeiro anualizado
# ────────────────────────────────────────────────────────────────────────
def plot_impacto_financeiro(resultados: list, path_fig):
    """Barras empilhadas do impacto anual projetado, F1 vs F2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Painel 1: Custos anuais por categoria
    ax = axes[0]
    familias = [r["familia"] for r in resultados]
    categorias = ["Venda perdida", "Viagens extras",
                  "Descarte", "Custo ruptura op."]
    dados = np.array([
        [r["impacto"]["venda_perdida_anual_R$"] for r in resultados],
        [r["impacto"]["custo_viagens_anual_R$"] for r in resultados],
        [r["descarte"]["custo_descarte_R$"] * 52 / N_TESTE for r in resultados],
        [r["impacto"]["custo_ruptura_anual_R$"] for r in resultados],
    ])

    cores_cat = ["#E74C3C", "#F39C12", "#9B59B6", "#3498DB"]
    bottom = np.zeros(len(familias))
    for i, (cat, cor) in enumerate(zip(categorias, cores_cat)):
        ax.bar(familias, dados[i], bottom=bottom, color=cor,
               label=cat, edgecolor="black", linewidth=0.5)
        # Valor dentro de cada segmento
        for j, val in enumerate(dados[i]):
            if val > dados.sum(axis=0)[j] * 0.05:  # só mostra se significativo
                ax.text(j, bottom[j] + val/2, f"R$ {val/1000:.0f}k",
                        ha="center", va="center",
                        color="white", fontweight="bold", fontsize=9)
        bottom += dados[i]

    # Totais no topo
    for j, total in enumerate(bottom):
        ax.text(j, total * 1.02, f"R$ {total/1000:.0f}k",
                ha="center", fontweight="bold", fontsize=11)

    ax.set_ylabel("Custo anual projetado (R$)", fontsize=11)
    ax.set_title("Impacto financeiro anual — política recomendada",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Painel 2: Comparação SS política atual vs recomendada
    ax = axes[1]
    ss_atual = []
    ss_recomendado = []
    for r in resultados:
        # Política atual: 2 semanas de cobertura
        demanda_media = float(np.mean(r["serie_full"]))
        ss_atual.append(demanda_media * 2)
        ss_recomendado.append(r["ss"])

    x = np.arange(len(familias))
    w = 0.35
    ax.bar(x - w/2, ss_atual, w, color="#E74C3C", alpha=0.8,
           label="Política atual (2 sem cobertura)",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, ss_recomendado, w, color="#2ECC71", alpha=0.8,
           label="Recomendado (z·RMSE·√LT)",
           edgecolor="black", linewidth=0.5)
    for i, (a, r) in enumerate(zip(ss_atual, ss_recomendado)):
        ax.text(i - w/2, a * 1.02, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + w/2, r * 1.02, f"{r:.0f}", ha="center", fontsize=9)
        delta = 100 * (r - a) / a
        ax.text(i, max(a, r) * 1.12, f"Δ {delta:+.0f}%",
                ha="center", fontweight="bold",
                color="green" if delta < 0 else "red", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(familias)
    ax.set_ylabel("Estoque de segurança (un)")
    ax.set_title("Estoque de segurança: atual vs recomendado",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.margins(y=0.2)

    fig.suptitle("Q9 — Impacto financeiro e dimensionamento de estoque",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(path_fig, bbox_inches="tight", dpi=150)
    plt.close()


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Q9 — Cadeia Causal: Erro de Previsão → Impacto Operacional")
    print("Par contrastante: F1 (Perecíveis) × F2 (Premium)")
    print("=" * 70)

    # 1. Carregar dados
    df_fam = load_demand_family()
    df_perfil = load_family_profile()
    df_params = load_simulation_params()
    print(f"\n[1/5] Dados carregados.")

    # 2. Para cada família (F1, F2), rodar cadeia causal
    resultados = []
    modelo_por_fam = {"F1": "Holt-Winters", "F2": "SES"}

    for familia in ["F1", "F2"]:
        print(f"\n[2/5] Processando {familia}...")
        serie = (df_fam[df_fam["familia_cod"] == familia]
                 .sort_values("semana")["demanda_total"].reset_index(drop=True))
        params_fam = df_params[df_params["familia_cod"] == familia].iloc[0].to_dict()
        perf_fam = df_perfil[df_perfil["familia_cod"] == familia].iloc[0].to_dict()

        # Elo 0: previsão
        pred, rmse, real_teste = previsao_melhor_modelo(serie, familia)
        print(f"       RMSE (modelo {modelo_por_fam[familia]}): {rmse:.1f} un/sem")

        # Elo 1: SS
        lt_dias = params_fam["lt_m"]
        z = Z_SCORE[familia]
        ss = elo_1_estoque_seguranca(rmse, lt_dias, z)
        print(f"       SS = {z:.3f} × {rmse:.0f} × √({lt_dias}/7) = {ss:.0f} un")

        # Elo 2: rupturas
        ruptura = elo_2_simular_rupturas(real_teste, pred, ss)
        print(f"       {ruptura['n_rupturas']} rupturas em {N_TESTE} sem "
              f"({ruptura['pct_semanas_ruptura']:.1f}%), "
              f"fill rate={ruptura['fill_rate_sim']:.1f}%")

        # Elo 3: custos operacionais
        impacto = elo_3_propagar_custos(ruptura, familia, params_fam, perf_fam, N_TESTE)
        print(f"       Venda perdida no teste: R$ {impacto['venda_perdida_teste_R$']:,.0f}")

        # Elo 4: descarte (só faz sentido para F1)
        sl_sem = perf_fam["sl_sem"]
        descarte = elo_4_descarte_pereciveis(pred, real_teste, params_fam, sl_sem)
        print(f"       Descarte estimado no teste: R$ {descarte['custo_descarte_R$']:,.0f}")

        resultados.append({
            "familia": familia,
            "modelo": modelo_por_fam[familia],
            "serie_full": serie.values,
            "real_teste": real_teste,
            "pred_teste": pred,
            "rmse": rmse,
            "lt_dias": lt_dias,
            "lt_sem": lt_dias / 7,
            "z": z,
            "ss": ss,
            "ruptura": ruptura,
            "impacto": impacto,
            "descarte": descarte,
            "params": params_fam,
        })

    # 3. Tabelas consolidadas
    print(f"\n[3/5] Consolidando tabelas...")
    linhas = []
    for r in resultados:
        linhas.append({
            "familia": r["familia"],
            "modelo": r["modelo"],
            "RMSE_un_sem": round(r["rmse"], 1),
            "LT_dias": r["lt_dias"],
            "z_nivel_servico": round(r["z"], 3),
            "SS_recomendado_un": round(r["ss"], 0),
            "rupturas_teste_n": r["ruptura"]["n_rupturas"],
            "pct_semanas_ruptura": round(r["ruptura"]["pct_semanas_ruptura"], 1),
            "un_faltantes": round(r["ruptura"]["un_faltantes"], 0),
            "fill_rate_simulado_%": round(r["ruptura"]["fill_rate_sim"], 1),
            "venda_perdida_teste_R$": round(r["impacto"]["venda_perdida_teste_R$"], 0),
            "custo_ruptura_teste_R$": round(r["impacto"]["custo_ruptura_teste_R$"], 0),
            "n_viagens_teste": r["impacto"]["n_viagens_extras_teste"],
            "custo_viagens_teste_R$": r["impacto"]["custo_viagens_teste_R$"],
            "co2_teste_kg": r["impacto"]["co2_viagens_teste_kg"],
            "descarte_teste_R$": round(r["descarte"]["custo_descarte_R$"], 0),
        })
    df_cadeia = pd.DataFrame(linhas)
    df_cadeia.to_csv(PATH_TABLES / "q9_cadeia_causal.csv",
                     index=False, encoding="utf-8-sig")

    # Tabela anualizada
    linhas_anual = []
    for r in resultados:
        linhas_anual.append({
            "familia": r["familia"],
            "venda_perdida_R$_ano": round(r["impacto"]["venda_perdida_anual_R$"], 0),
            "custo_ruptura_R$_ano": round(r["impacto"]["custo_ruptura_anual_R$"], 0),
            "viagens_extras_n_ano": round(r["impacto"]["n_viagens_anual"], 1),
            "custo_viagens_R$_ano": round(r["impacto"]["custo_viagens_anual_R$"], 0),
            "co2_kg_ano": round(r["impacto"]["co2_viagens_anual_kg"], 0),
            "descarte_R$_ano": round(r["descarte"]["custo_descarte_R$"] * 52 / N_TESTE, 0),
        })
    df_anual = pd.DataFrame(linhas_anual)
    df_anual["TOTAL_R$_ano"] = (df_anual["venda_perdida_R$_ano"] +
                                 df_anual["custo_ruptura_R$_ano"] +
                                 df_anual["custo_viagens_R$_ano"] +
                                 df_anual["descarte_R$_ano"])
    df_anual.to_csv(PATH_TABLES / "q9_impacto_anual.csv",
                    index=False, encoding="utf-8-sig")

    # 4. Figuras
    print(f"\n[4/5] Gerando figuras...")
    plot_cadeia_causal(resultados, PATH_FIGURES / "q9_cadeia_causal.png")
    plot_impacto_financeiro(resultados, PATH_FIGURES / "q9_impacto_financeiro.png")

    # 5. Resumo
    print(f"\n[5/5] Resumo:")
    print("\n• Cadeia causal (período de teste, 24 semanas):")
    print(df_cadeia.to_string(index=False))
    print("\n• Impacto anual projetado:")
    print(df_anual.to_string(index=False))

    print(f"\n✅ Saídas geradas em outputs/tables/ e outputs/figures/")


if __name__ == "__main__":
    main()
