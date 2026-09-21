"""Carregamento e validação da versão horária do Bike Sharing (UCI).

Pandas é usado aqui somente para leitura, seleção e transformação dos dados.
Os cálculos estatísticos exibidos pela aplicação ficam no núcleo estatístico.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


FONTE_URL = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DOI = "10.24432/C5W894"
CAMINHO_PADRAO = Path(__file__).resolve().parent / "dados" / "dataset.csv"
TOTAL_REGISTROS = 17379
COLUNAS_ORIGINAIS = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
)

NUMERICAS = {
    "temp": "Temperatura (normalizada)",
    "atemp": "Sensação térmica (normalizada)",
    "hum": "Umidade (normalizada)",
    "windspeed": "Velocidade do vento (normalizada)",
    "casual": "Aluguéis de usuários casuais",
    "registered": "Aluguéis de usuários cadastrados",
    "cnt": "Total de aluguéis por hora",
}

CATEGORICAS = {
    "clima": "Condição do tempo",
    "tipo_dia": "Tipo de dia",
    "ano": "Ano",
    "feriado": "Feriado",
}

CLIMAS = {
    1: "1 — Limpo ou poucas nuvens",
    2: "2 — Névoa ou muitas nuvens",
    3: "3 — Chuva/neve leve",
    4: "4 — Chuva/neve forte",
}

DOMINIOS = {
    "season": {1, 2, 3, 4},
    "yr": {0, 1},
    "mnth": set(range(1, 13)),
    "hr": set(range(24)),
    "holiday": {0, 1},
    "weekday": set(range(7)),
    "workingday": {0, 1},
    "weathersit": {1, 2, 3, 4},
}


def carregar_dados(caminho: str | Path | None = None) -> pd.DataFrame:
    """Leia o CSV original, valide-o e acrescente rótulos em português.

    O laboratório foi preparado para o arquivo completo ``hour.csv`` de
    17.379 observações. Um arquivo incompleto ou de outro esquema é rejeitado
    explicitamente; nenhuma linha é descartada, imputada ou criada.
    """
    arquivo = Path(caminho) if caminho is not None else CAMINHO_PADRAO
    if not arquivo.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado: {arquivo}. "
            "Na pasta do projeto, execute: python scripts/baixar_dados.py"
        )
    try:
        quadro = pd.read_csv(arquivo)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as erro:
        raise ValueError(f"Não foi possível ler o CSV de dados: {erro}") from erro

    encontradas = set(quadro.columns)
    esperadas = set(COLUNAS_ORIGINAIS)
    if encontradas != esperadas:
        faltantes = sorted(esperadas - encontradas)
        extras = sorted(encontradas - esperadas)
        raise ValueError(
            "Esquema do dataset inválido. "
            f"Colunas faltantes: {faltantes}; colunas extras: {extras}. "
            "Use a versão horária original (hour.csv)."
        )
    if len(quadro) != TOTAL_REGISTROS:
        raise ValueError(
            f"Quantidade de registros inválida: {len(quadro):,}. "
            f"O hour.csv completo deve conter {TOTAL_REGISTROS:,} linhas."
        )
    if quadro.isna().any().any():
        colunas = quadro.columns[quadro.isna().any()].tolist()
        raise ValueError(f"O dataset contém valores ausentes nas colunas: {colunas}.")

    for coluna in COLUNAS_ORIGINAIS:
        if coluna == "dteday":
            continue
        try:
            numeros = pd.to_numeric(quadro[coluna], errors="raise")
        except (ValueError, TypeError) as erro:
            raise ValueError(f"A coluna '{coluna}' contém valor não numérico.") from erro
        if not all(math.isfinite(float(valor)) for valor in numeros):
            raise ValueError(f"A coluna '{coluna}' contém valor não finito.")
        quadro[coluna] = numeros

    inteiras = ["instant", *DOMINIOS, "casual", "registered", "cnt"]
    for coluna in inteiras:
        if any(float(valor) % 1 != 0 for valor in quadro[coluna]):
            raise ValueError(f"A coluna '{coluna}' deve conter somente números inteiros.")
        quadro[coluna] = quadro[coluna].astype("int64")

    for coluna, dominio in DOMINIOS.items():
        invalidos = sorted(set(quadro[coluna]) - dominio)
        if invalidos:
            raise ValueError(f"Códigos inválidos na coluna '{coluna}': {invalidos}.")

    for coluna in ("temp", "atemp", "hum", "windspeed"):
        if not quadro[coluna].between(0, 1).all():
            raise ValueError(f"A coluna normalizada '{coluna}' deve estar entre 0 e 1.")
    for coluna in ("casual", "registered", "cnt"):
        if (quadro[coluna] < 0).any():
            raise ValueError(f"A coluna '{coluna}' contém contagem negativa.")
    if not (quadro["cnt"] == quadro["casual"] + quadro["registered"]).all():
        raise ValueError("Contagens inconsistentes: cnt deve ser igual a casual + registered.")
    if (quadro["instant"] <= 0).any() or quadro["instant"].duplicated().any():
        raise ValueError("O identificador 'instant' deve ser positivo e único.")

    try:
        quadro["dteday"] = pd.to_datetime(quadro["dteday"], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as erro:
        raise ValueError("A coluna 'dteday' contém data inválida; esperado AAAA-MM-DD.") from erro
    if not (quadro["dteday"].dt.year == quadro["yr"] + 2011).all():
        raise ValueError("O ano em 'dteday' não corresponde ao código em 'yr'.")
    if not (quadro["dteday"].dt.month == quadro["mnth"]).all():
        raise ValueError("O mês em 'dteday' não corresponde ao código em 'mnth'.")
    if not (((quadro["dteday"].dt.dayofweek + 1) % 7) == quadro["weekday"]).all():
        raise ValueError("O dia da semana em 'dteday' não corresponde a 'weekday'.")
    if quadro.duplicated(subset=["dteday", "hr"]).any():
        raise ValueError("O dataset contém observações duplicadas para a mesma data e hora.")

    quadro["clima"] = quadro["weathersit"].map(CLIMAS)
    quadro["tipo_dia"] = quadro["workingday"].map({0: "Fim de semana ou feriado", 1: "Dia útil"})
    quadro["ano"] = quadro["yr"].map({0: "2011", 1: "2012"})
    quadro["feriado"] = quadro["holiday"].map({0: "Não", 1: "Sim"})
    return quadro
