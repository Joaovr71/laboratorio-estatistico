"""Testes de integração da interface e dos dados enviados aos gráficos.

Execute na raiz: python -m pytest test_app.py -q
AppTest verifica a aplicação real e suas interações, sem abrir um navegador.
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import minhastats as ms
from dados import CATEGORICAS, carregar_dados


RAIZ = Path(__file__).resolve().parent


@pytest.fixture
def app(tmp_path, monkeypatch):
    # Evita depender de escrita no cache global do usuário em ambientes restritos.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    instancia = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60).run()
    sem_erros(instancia)
    return instancia


def sem_erros(app):
    assert not app.exception, [erro.message for erro in app.exception]
    assert not app.error, [erro.value for erro in app.error]


def abrir_modulo(app, indice):
    seletor = app.sidebar.radio[0]
    app = seletor.set_value(seletor.options[indice]).run()
    sem_erros(app)
    return app


@pytest.mark.parametrize("indice,graficos_minimos", [(0, 0), (1, 0), (2, 2), (3, 2), (4, 1), (5, 1), (6, 3)])
def test_abre_os_sete_modulos_com_conteudo(app, indice, graficos_minimos):
    app = abrir_modulo(app, indice)
    assert app.header[0].value == app.sidebar.radio[0].value
    assert len(app.get("image")) >= graficos_minimos
    if indice == 0:
        assert app.metric[0].value == "17.379"
        assert len(app.dataframe) == 2
    elif indice == 1:
        medidas = app.dataframe[0].value
        linha = medidas.loc[medidas["Medida"] == "media", "Resultado"].iloc[0]
        esperado = ms.media(carregar_dados()["cnt"].tolist())
        assert float(linha.replace(",", "")) == pytest.approx(esperado, abs=1e-5)
    elif indice == 2:
        assert app.dataframe[0].value["frequencia"].sum() == 17379
    elif indice == 5:
        assert any("não implica causalidade" in aviso.value for aviso in app.info)
    elif indice == 6:
        assert len(app.subheader) == 3
        assert len(app.get("download_button")) >= 1


def test_exploracao_categorica_muda_variavel_e_preserva_contagens(app):
    app = abrir_modulo(app, 2)
    app.main.radio[0].set_value("Categórica").run()
    sem_erros(app)
    app.selectbox[0].set_value(list(CATEGORICAS)[1]).run()
    sem_erros(app)
    tabela = app.dataframe[0].value
    assert "categoria" in tabela.columns
    assert tabela["frequencia"].sum() == 17379
    assert tabela["relativa (%)"].sum() == pytest.approx(100)
    assert tabela["acumulada"].iloc[-1] == 17379
    assert len(app.get("image")) == 1


def test_distribuicao_exponencial_de_contagens_exibe_parametro_e_limite(app):
    app = abrir_modulo(app, 4)
    app.selectbox[0].set_value("cnt").run()
    sem_erros(app)
    app.selectbox[1].set_value("Exponencial").run()
    sem_erros(app)
    assert any("Taxa Exponencial" in bloco.value for bloco in app.markdown)
    assert any("contagens discretas" in aviso.value for aviso in app.warning)
    assert len(app.get("image")) == 1


def test_regressao_extrapolacao_e_predicao_coerente(app):
    app = abrir_modulo(app, 5)
    dados = carregar_dados()
    x, y = dados["temp"].tolist(), dados["cnt"].tolist()
    b0, b1, _ = ms.regressao_linear(x, y)
    valor = max(x) + 1
    app.number_input[0].set_value(valor).run()
    sem_erros(app)
    assert any("Extrapolação" in aviso.value for aviso in app.warning)
    resultado = next(item.value for item in app.metric if item.label == "Predição pela reta")
    assert float(resultado) == pytest.approx(b0 + b1 * valor, abs=1e-4)
    app.number_input[0].set_value(-100.0).run()
    sem_erros(app)
    assert any("contagem negativa" in aviso.value for aviso in app.warning)


def test_regressao_troca_variaveis_e_adverte_associacao_parte_total(app):
    app = abrir_modulo(app, 5)
    app.selectbox(key="x").set_value("casual").run()
    sem_erros(app)
    app.selectbox(key="y").set_value("cnt").run()
    sem_erros(app)
    assert any("associação estrutural" in aviso.value for aviso in app.warning)
    app.selectbox(key="x").set_value("cnt").run()
    sem_erros(app)
    assert app.selectbox(key="x").value != app.selectbox(key="y").value


def test_simulacoes_respondem_a_sliders_e_semente(app):
    app = abrir_modulo(app, 3)
    app.number_input[0].set_value(17)
    app.slider[0].set_value(100)
    app.slider[1].set_value(2)
    app.slider[2].set_value(100)
    app.run()
    sem_erros(app)
    esperado = ms.simular_lgn(100, seed=17)[-1]
    assert any(f"**{esperado:.4f}**" in bloco.value and "Frequência final" in bloco.value for bloco in app.markdown)
    assert [controle.value for controle in app.slider] == [100, 2, 100]
    assert len(app.get("image")) == 2
    app.slider[0].set_value(20000)
    app.slider[1].set_value(150)
    app.slider[2].set_value(3000)
    app.selectbox[0].set_value("windspeed")
    app.run()
    sem_erros(app)
    assert [controle.value for controle in app.slider] == [20000, 150, 3000]
    assert any("Média das médias" in bloco.value for bloco in app.markdown)


def test_graficos_usam_frequencias_e_boxplot_precalculados(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.axes
    import matplotlib.cbook
    import matplotlib.pyplot as plt
    import graficos as g

    def estatistica_automatica_proibida(*args, **kwargs):
        pytest.fail("O gráfico tentou calcular automaticamente estatísticas fora de minhastats.")

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", estatistica_automatica_proibida)
    monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", estatistica_automatica_proibida)
    monkeypatch.setattr(matplotlib.cbook, "boxplot_stats", estatistica_automatica_proibida)
    dados = [0, 1, 2, 2, 2, 3, 5, 40]
    figuras = []
    try:
        fig, ax = g.histograma(dados, "Teste", densidade=True)
        figuras.append(fig)
        tabela = ms.frequencias(dados)
        np.testing.assert_allclose([barra.get_height() for barra in ax.patches],
                                   [linha["relativa"] / (linha["superior"] - linha["inferior"]) for linha in tabela])
        assert sum(barra.get_height() * barra.get_width() for barra in ax.patches) == pytest.approx(1)
        recebidos = []
        bxp_original = matplotlib.axes.Axes.bxp

        def observar_bxp(self, bxpstats, *args, **kwargs):
            recebidos.extend(bxpstats)
            return bxp_original(self, bxpstats, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "bxp", observar_bxp)
        figuras.append(g.boxplot(dados, "Teste"))
        o = ms.outliers_iqr(dados)
        assert recebidos == [{"med": ms.mediana(dados), "q1": o["q1"], "q3": o["q3"],
                              "whislo": o["bigode_inferior"], "whishi": o["bigode_superior"], "fliers": o["outliers"]}]
        x, y = [1, 2, 3, 4], [2, 5, 5, 9]
        fig = g.regressao(x, y, "X", "Y")
        figuras.append(fig)
        b0, b1, _ = ms.regressao_linear(x, y)
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [b0 + b1 * min(x), b0 + b1 * max(x)])
        assert len(fig.axes[0].collections[0].get_offsets()) == len(x)
    finally:
        for fig in figuras:
            plt.close(fig)
