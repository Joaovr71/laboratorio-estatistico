"""Validação independente do núcleo contra NumPy e SciPy.

Execute: python -m pytest -q
Tolerância numérica documentada: rtol=1e-9, atol=1e-10.
As bibliotecas de referência são usadas exclusivamente na validação.
"""

import ast
import inspect
import math
import random

import numpy as np
import pytest
from scipy import stats

import minhastats as ms


RTOL, ATOL = 1e-9, 1e-10
AMOSTRAS = [
    [3, 1, 7, 2, 9],
    [-8, -1, 0, 2, 4, 100],
    [2, 2, 2, 2],
    [0.1, 0.2, 0.3, 0.4, 1.5, 2.7],
    list(np.random.default_rng(31415).normal(20, 4, 503)),
    [1e12 + v for v in (-5, -2, 0, 1, 6)],
]


def igual(obtido, esperado):
    np.testing.assert_allclose(obtido, esperado, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("dados", AMOSTRAS)
def test_tendencia_central_amplitude_e_quartis(dados):
    igual(ms.media(dados), np.mean(dados))
    igual(ms.mediana(dados), np.median(dados))
    igual(ms.amplitude(dados), np.ptp(dados))
    igual(ms.quartis(dados), np.percentile(dados, [25, 50, 75], method="linear"))
    valores, contagens = np.unique(dados, return_counts=True)
    igual(ms.moda(dados), valores[contagens == contagens.max()])
    # SciPy retorna uma só moda no empate; ela pertence às modas autorais.
    assert float(stats.mode(dados, keepdims=False).mode) in ms.moda(dados)


@pytest.mark.parametrize("dados", AMOSTRAS)
@pytest.mark.parametrize("p", [0, 1, 10, 25, 33.333, 50, 75, 99, 100])
def test_percentis_interpolados(dados, p):
    igual(ms.percentil(dados, p), np.percentile(dados, p, method="linear"))


@pytest.mark.parametrize("dados", AMOSTRAS)
@pytest.mark.parametrize("amostral", [True, False])
def test_dispersao(dados, amostral):
    ddof = int(amostral)
    igual(ms.variancia(dados, amostral), np.var(dados, ddof=ddof))
    igual(ms.desvio_padrao(dados, amostral), np.std(dados, ddof=ddof))
    igual(ms.coeficiente_variacao(dados, amostral), abs(stats.variation(dados, ddof=ddof)) * 100)


@pytest.mark.parametrize("dados", [[-4, -3, -2], [0, 1, 9, 1e4], [1]])
def test_cv_populacional_inclui_media_negativa(dados):
    igual(ms.coeficiente_variacao(dados, False), abs(stats.variation(dados)) * 100)


@pytest.mark.parametrize("dados", [[0, 0], [-1, 1], [-1, 1 + 1e-14]])
def test_cv_media_zero_ou_proxima(dados):
    with pytest.raises(ValueError, match="média"):
        ms.coeficiente_variacao(dados)


PARES = [
    ([1, 2, 3, 4], [8, 6, 4, 2]),
    ([-2, -1, 0, 1, 2], [2, 2.5, 3, 3.5, 4]),
    ([2, 8, 4, 9, 1, 7], [8, 2, 5, 4, 9, 1]),
    (list(np.random.default_rng(17).uniform(-10, 50, 300)),
     list(np.random.default_rng(22).normal(10, 3, 300))),
]


@pytest.mark.parametrize("x,y", PARES)
@pytest.mark.parametrize("amostral", [True, False])
def test_covariancia(x, y, amostral):
    igual(ms.covariancia(x, y, amostral), np.cov(x, y, ddof=int(amostral))[0, 1])


@pytest.mark.parametrize("x,y", PARES)
def test_correlacao_e_regressao(x, y):
    igual(ms.correlacao(x, y), stats.pearsonr(x, y).statistic)
    igual(ms.correlacao(x, y), np.corrcoef(x, y)[0, 1])
    referencia = stats.linregress(x, y)
    b0, b1, r2 = ms.regressao_linear(x, y)
    igual([b0, b1, r2], [referencia.intercept, referencia.slope, referencia.rvalue ** 2])
    residuos = np.asarray(y) - (b0 + b1 * np.asarray(x))
    # Identidade dos mínimos quadrados: resíduos ortogonais a constante e X.
    igual(np.sum(residuos), 0)
    igual(np.dot(residuos, np.asarray(x) - np.mean(x)), 0)


def test_regressao_com_deslocamento_grande():
    x = [1e12 + i for i in range(1, 9)]
    y = [3 * i + 7 for i in range(1, 9)]
    referencia = stats.linregress(x, y)
    igual(ms.regressao_linear(x, y), [referencia.intercept, referencia.slope, referencia.rvalue ** 2])


@pytest.mark.parametrize("funcao", [ms.covariancia, ms.correlacao, ms.regressao_linear])
def test_vetores_incompativeis(funcao):
    with pytest.raises(ValueError, match="mesmo tamanho"):
        funcao([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        funcao([1], [2])


@pytest.mark.parametrize("funcao", [ms.correlacao, ms.regressao_linear])
@pytest.mark.parametrize("x,y", [([1, 1], [2, 3]), ([1, 2], [3, 3]), ([1, 1], [1, 1])])
def test_constantes_sem_correlacao_definida(funcao, x, y):
    with pytest.raises(ValueError, match="constante"):
        funcao(x, y)


def test_covariancia_constante_e_unitaria_populacional():
    igual(ms.covariancia([1, 1], [2, 3]), np.cov([1, 1], [2, 3])[0, 1])
    assert ms.covariancia([1], [3], False) == 0


@pytest.mark.parametrize("dados", AMOSTRAS)
def test_resumo(dados):
    resumo = ms.resumo(dados)
    assert resumo["n"] == len(dados)
    igual([resumo["minimo"], resumo["maximo"], resumo["media"], resumo["mediana"], resumo["amplitude"]],
          [np.min(dados), np.max(dados), np.mean(dados), np.median(dados), np.ptp(dados)])
    igual([resumo["q1"], resumo["q2"], resumo["q3"]], np.percentile(dados, [25, 50, 75]))
    igual(resumo["variancia_amostral"], np.var(dados, ddof=1))
    igual(resumo["variancia_populacional"], np.var(dados))
    igual(resumo["desvio_padrao_amostral"], np.std(dados, ddof=1))
    igual(resumo["desvio_padrao_populacional"], np.std(dados))
    igual(resumo["coeficiente_variacao"], abs(stats.variation(dados, ddof=1)) * 100)
    assert resumo["moda"] == ms.moda(dados)


def test_uma_observacao_e_resumo_indefinido():
    assert ms.media([7]) == ms.mediana([7]) == 7
    assert ms.moda([7]) == [7]
    assert ms.quartis([7]) == (7, 7, 7)
    assert ms.variancia([7], False) == ms.desvio_padrao([7], False) == 0
    resultado = ms.resumo([7])
    for chave in ["variancia_amostral", "desvio_padrao_amostral", "coeficiente_variacao"]:
        assert resultado[chave] is None
    for funcao in [ms.variancia, ms.desvio_padrao, ms.coeficiente_variacao]:
        with pytest.raises(ValueError):
            funcao([7])
    assert ms.resumo([-1, 1])["coeficiente_variacao"] is None


@pytest.mark.parametrize("dados", [list(range(21)), [-8, -1, 0, 2, 4, 100], [3, 3, 3], [0.1, 0.2, 0.3, 0.4, 0.5]])
@pytest.mark.parametrize("classes", [None, 1, 3, 7])
def test_frequencias_numericas_contra_histograma_numpy(dados, classes):
    tabela = ms.frequencias(dados, classes)
    k = classes or (1 if min(dados) == max(dados) else math.ceil(1 + math.log2(len(dados))))
    contagens, bordas = np.histogram(dados, bins=k)
    igual([r["frequencia"] for r in tabela], contagens)
    igual([r["inferior"] for r in tabela] + [tabela[-1]["superior"]], bordas)
    igual([r["relativa"] for r in tabela], contagens / len(dados))
    igual([r["acumulada"] for r in tabela], np.cumsum(contagens))
    assert tabela[-1]["acumulada"] == len(dados)


def test_bordas_histograma_inclui_maximo():
    tabela = ms.frequencias([0, 1, 2, 3, 4], 4)
    assert [r["frequencia"] for r in tabela] == [1, 1, 1, 2]
    assert ms.frequencias([1], 1)[0]["frequencia"] == 1


def test_frequencias_categoricas_contra_numpy():
    dados = ["verão", "outono", "verão", "inverno", "outono", "verão"]
    tabela = ms.frequencias_categoricas(dados)
    valores, indices, contagens = np.unique(dados, return_index=True, return_counts=True)
    ordem = np.argsort(indices)
    assert [r["categoria"] for r in tabela] == list(valores[ordem])
    igual([r["frequencia"] for r in tabela], contagens[ordem])
    igual([r["relativa"] for r in tabela], contagens[ordem] / len(dados))
    igual([r["acumulada"] for r in tabela], np.cumsum(contagens[ordem]))


@pytest.mark.parametrize("dados", AMOSTRAS + [[1], [1, 1, 1, 1, 1, 20], [-99, 0, 1, 2, 3, 99]])
def test_outliers_iqr_contra_numpy(dados):
    resultado = ms.outliers_iqr(dados)
    q1, q3 = np.percentile(dados, [25, 75], method="linear")
    iqr = stats.iqr(dados, interpolation="linear")
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    valores = np.asarray(dados)
    mascara = (valores < inferior) | (valores > superior)
    igual([resultado["q1"], resultado["q3"], resultado["iqr"], resultado["limite_inferior"], resultado["limite_superior"]],
          [q1, q3, iqr, inferior, superior])
    igual(resultado["outliers"], valores[mascara])
    igual([resultado["bigode_inferior"], resultado["bigode_superior"]],
          [np.min(valores[~mascara]), np.max(valores[~mascara])])


@pytest.mark.parametrize("dados,trecho", [([1, 1, 1], "iguais"), ([1, 2, 3], "próximas"),
                                         ([0, 0, 1, 1, 100], "direita"), ([-100, 0, 0, 1, 1], "esquerda")])
def test_interpretacao_informa_limites_e_outliers(dados, trecho):
    texto = ms.interpretacao(dados)
    assert trecho in texto
    assert "heurística" in texto
    assert "não foram removidos" in texto
    assert f"{len(ms.outliers_iqr(dados)['outliers'])} de {len(dados)}" in texto


@pytest.mark.parametrize("x", [-20, -1, 0, 0.5, 1, 3, 20])
@pytest.mark.parametrize("mu,sigma", [(0, 1), (2, 3), (-3, 0.25)])
def test_normal_contra_scipy(x, mu, sigma):
    igual(ms.normal_pdf(x, mu, sigma), stats.norm.pdf(x, loc=mu, scale=sigma))


@pytest.mark.parametrize("x", [-20, -2, -1, 0, 1, 4, 20])
def test_uniforme_contra_scipy(x):
    igual(ms.uniforme_pdf(x, -2, 4), stats.uniform.pdf(x, loc=-2, scale=6))


@pytest.mark.parametrize("x", [-1, 0, 0.25, 1, 4, 100])
@pytest.mark.parametrize("taxa", [0.1, 1, 3])
def test_exponencial_contra_scipy(x, taxa):
    igual(ms.exponencial_pdf(x, taxa), stats.expon.pdf(x, scale=1 / taxa))


def test_lgn_contra_referencia_numpy_e_reprodutibilidade():
    rng = random.Random(73)
    lancamentos = np.array([int(rng.random() < 0.5) for _ in range(2000)])
    esperado = np.cumsum(lancamentos) / np.arange(1, 2001)
    igual(ms.simular_lgn(2000, seed=73), esperado)
    assert ms.simular_lgn(20, seed=73) == ms.simular_lgn(20, seed=73)
    assert ms.simular_lgn(20, seed=73) != ms.simular_lgn(20, seed=74)


def test_tcl_contra_medias_numpy_e_amostragem_com_reposicao():
    dados, tamanho, repeticoes, seed = [1, 5, 9], 12, 80, 27
    rng = random.Random(seed)
    amostras = np.array([[rng.choice(dados) for _ in range(tamanho)] for _ in range(repeticoes)])
    igual(ms.simular_tcl(dados, tamanho, repeticoes, seed), np.mean(amostras, axis=1))
    assert ms.simular_tcl([7, 7], 50, 4) == [7] * 4
    assert ms.simular_tcl(dados, 1, 20, seed) == ms.simular_tcl(dados, 1, 20, seed)


def test_simulacoes_preservam_gerador_global():
    estado = random.getstate()
    ms.simular_lgn(10)
    ms.simular_tcl([1, 2, 3], 4, 5)
    assert random.getstate() == estado


FUNCOES_UNIVARIADAS = [ms.media, ms.mediana, ms.moda, ms.amplitude, ms.quartis,
                      ms.variancia, ms.desvio_padrao, ms.coeficiente_variacao,
                      ms.resumo, ms.frequencias, ms.outliers_iqr, ms.interpretacao,
                      lambda x: ms.percentil(x, 50), lambda x: ms.simular_tcl(x, 2, 3)]


@pytest.mark.parametrize("funcao", FUNCOES_UNIVARIADAS)
@pytest.mark.parametrize("dados", [[], [1, float("nan")], [float("inf")], [-float("inf")], ["1", "2"], [True, False], None])
def test_amostras_invalidas(funcao, dados):
    with pytest.raises(ValueError):
        funcao(dados)


@pytest.mark.parametrize("p", [-1, 101, float("nan"), float("inf"), "50", True])
def test_percentil_invalido(p):
    with pytest.raises(ValueError):
        ms.percentil([1, 2, 3], p)


@pytest.mark.parametrize("valor", [0, -1, 1.5, float("nan"), float("inf"), "3", True])
def test_contagens_e_classes_invalidas(valor):
    for executar in [lambda: ms.frequencias([1, 2], valor), lambda: ms.simular_lgn(valor),
                     lambda: ms.simular_tcl([1, 2], valor, 3), lambda: ms.simular_tcl([1, 2], 3, valor)]:
        with pytest.raises(ValueError):
            executar()


@pytest.mark.parametrize("dados", [[], [None], [float("nan")], [float("inf")], [[1, 2]], None])
def test_categorias_invalidas(dados):
    with pytest.raises(ValueError):
        ms.frequencias_categoricas(dados)


@pytest.mark.parametrize("executar", [
    lambda: ms.normal_pdf(0, 0, 0), lambda: ms.normal_pdf(0, 0, -1),
    lambda: ms.normal_pdf(float("nan"), 0, 1), lambda: ms.normal_pdf(0, float("inf"), 1),
    lambda: ms.uniforme_pdf(0, 1, 1), lambda: ms.uniforme_pdf(0, 2, 1),
    lambda: ms.uniforme_pdf(0, 1, float("inf")),
    lambda: ms.exponencial_pdf(0, 0), lambda: ms.exponencial_pdf(0, -1),
    lambda: ms.exponencial_pdf(float("inf"), 1),
    lambda: ms.variancia([1, 2], amostral="sim"),
    lambda: ms.covariancia([1, 2], [1, 2], amostral=1),
])
def test_parametros_invalidos(executar):
    with pytest.raises(ValueError):
        executar()


def test_medidas_nao_modificam_dados_e_aceitam_iteradores():
    dados = [3, 1, 2, 2]
    original = dados.copy()
    for funcao in FUNCOES_UNIVARIADAS:
        funcao(dados)
    assert dados == original
    assert ms.media(iter(dados)) == 2
    assert ms.mediana(iter(dados)) == 2


def test_estabilidade_soma_e_mediana_extremas():
    assert ms.media([1e16, 1, -1e16]) == 1 / 3
    assert ms.media([1e308, 1e308]) == 1e308
    assert ms.mediana([-1e308, 1e308]) == 0


def test_nucleo_importa_apenas_biblioteca_padrao_permitida():
    arvore = ast.parse(inspect.getsource(ms))
    modulos = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            modulos.update(alias.name.split(".")[0] for alias in no.names)
        elif isinstance(no, ast.ImportFrom):
            modulos.add(no.module.split(".")[0])
    assert modulos <= {"math", "random", "bisect"}
