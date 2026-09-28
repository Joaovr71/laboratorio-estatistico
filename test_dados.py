"""Integridade da base real e rejeição explícita de arquivos danificados."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from dados import CAMINHO_PADRAO, CATEGORICAS, COLUNAS_ORIGINAIS, NUMERICAS, carregar_dados


@pytest.fixture(scope="module")
def original():
    return pd.read_csv(CAMINHO_PADRAO)


def test_base_real_e_variaveis():
    quadro = carregar_dados()
    assert len(quadro) == 17379
    assert set(COLUNAS_ORIGINAIS) <= set(quadro.columns)
    assert len(NUMERICAS) >= 4
    assert len(CATEGORICAS) >= 2
    assert set(NUMERICAS) | set(CATEGORICAS) <= set(quadro.columns)
    assert set(quadro["ano"]) == {"2011", "2012"}
    assert set(quadro["tipo_dia"]) == {"Dia útil", "Fim de semana ou feriado"}
    assert quadro["cnt"].tolist() == (quadro["casual"] + quadro["registered"]).tolist()


def test_bytes_conferem_com_manifesto():
    manifesto = json.loads((CAMINHO_PADRAO.parent / "metadados.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(CAMINHO_PADRAO.read_bytes()).hexdigest() == manifesto["sha256_csv"]
    assert manifesto["registros"] == 17379
    assert manifesto["numero_colunas"] == 17


def test_arquivo_ausente_tem_orientacao(tmp_path):
    with pytest.raises(FileNotFoundError, match="baixar_dados.py"):
        carregar_dados(tmp_path / "ausente.csv")


def test_esquema_errado_rejeitado(original, tmp_path):
    arquivo = tmp_path / "incompleto.csv"
    original.drop(columns="cnt").to_csv(arquivo, index=False)
    with pytest.raises(ValueError, match="Colunas faltantes.*cnt"):
        carregar_dados(arquivo)


def test_linha_ausente_nao_e_imputada(original, tmp_path):
    arquivo = tmp_path / "sem_uma_linha.csv"
    original.iloc[:-1].to_csv(arquivo, index=False)
    with pytest.raises(ValueError, match="Quantidade de registros"):
        carregar_dados(arquivo)


@pytest.mark.parametrize(
    "coluna,valor,mensagem",
    [
        ("temp", None, "valores ausentes"),
        ("temp", float("inf"), "não finito"),
        ("temp", "texto", "não numérico"),
        ("temp", 1.1, "entre 0 e 1"),
        ("weathersit", 9, "Códigos inválidos.*weathersit"),
        ("hr", 1.5, "números inteiros"),
        ("casual", -1, "contagem negativa"),
        ("cnt", 999999, "cnt deve ser igual"),
        ("dteday", "data inválida", "data inválida"),
        ("yr", 1, "ano.*não corresponde"),
        ("instant", 0, "positivo e único"),
    ],
)
def test_valores_corrompidos_rejeitados(original, tmp_path, coluna, valor, mensagem):
    quadro = original.copy()
    # A corrupção deve chegar ao CSV, mesmo quando o pandas rejeitaria
    # atribuir texto ou uma fração diretamente a uma coluna inteira.
    quadro[coluna] = quadro[coluna].astype(object)
    quadro.loc[0, coluna] = valor
    arquivo = tmp_path / "corrompido.csv"
    quadro.to_csv(arquivo, index=False)
    with pytest.raises(ValueError, match=mensagem):
        carregar_dados(arquivo)


def test_identificador_duplicado_rejeitado(original, tmp_path):
    quadro = original.copy()
    quadro.loc[1, "instant"] = quadro.loc[0, "instant"]
    arquivo = tmp_path / "duplicado.csv"
    quadro.to_csv(arquivo, index=False)
    with pytest.raises(ValueError, match="positivo e único"):
        carregar_dados(arquivo)


def test_data_hora_duplicada_rejeitada(original, tmp_path):
    quadro = original.copy()
    quadro.loc[1, "hr"] = quadro.loc[0, "hr"]
    arquivo = tmp_path / "hora_duplicada.csv"
    quadro.to_csv(arquivo, index=False)
    with pytest.raises(ValueError, match="mesma data e hora"):
        carregar_dados(arquivo)
