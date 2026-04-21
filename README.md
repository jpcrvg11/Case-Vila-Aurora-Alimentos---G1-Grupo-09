# Vila Aurora Alimentos — Projeto Integrado PI5 (ENG4550)

## Contexto

Este repositório reúne a análise quantitativa desenvolvida pelo Grupo 09 para o caso **Vila Aurora Alimentos**, no âmbito da disciplina **ENG4550 — Projeto Integrado 5** do curso de Engenharia Industrial da **PUC-Rio** (2026.1). O trabalho cobre as entregas parciais **Q1 a Q9**, abrangendo diagnóstico de demanda, benchmarking de SKUs, classificação ABC/XYZ, modelagem de previsão e cadeia causal de indicadores.

---

## Estrutura de pastas

```
.
├── data/
│   └── raw/          Dados originais do case (read-only — não modificar)
├── src/              Scripts Python da análise
├── outputs/
│   ├── figures/      Gráficos gerados (.png) — versionados
│   └── tables/       Tabelas de resultados e Anexo F preenchido — versionados
├── reports/          Relatórios .docx
├── slides/           Apresentação final
├── requirements.txt  Dependências Python
└── README.md
```

---

## Como rodar

### 1. Criar e ativar o ambiente virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar os scripts (ordem recomendada)

```bash
python src/q4_benchmarking_sku.py
python src/q8_classificacao_e_modelos.py
python src/q9_cadeia_causal.py
```

Os arquivos gerados serão salvos em `outputs/figures/` e `outputs/tables/`.

---

## Mapeamento Questão → Script

| Questão | Script | Descrição |
|---------|--------|-----------|
| Q4 | `src/q4_benchmarking_sku.py` | Benchmarking de SKUs (baseado no Apêndice A do relatório) |
| Q8 | `src/q8_classificacao_e_modelos.py` | Classificação ABC/XYZ e modelos de previsão de demanda |
| Q9 | `src/q9_cadeia_causal.py` | Cadeia causal de indicadores operacionais |

---

## Autores

| Nome | Matrícula |
|------|-----------|
| João Pedro Leite de Almeida | [adicionar] |
| Lucas Agostinho | [adicionar] |
| Gabriel Stadler | [adicionar] |
| Felipe Sarmento | [adicionar] |

**Disciplina:** ENG4550 — Projeto Integrado 5
**Professor:** Rodrigo Caiado
**Período:** 2026.1 — PUC-Rio
