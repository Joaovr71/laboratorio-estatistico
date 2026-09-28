"""Reproduza validação, descobertas, figuras e RELATORIO.md a partir da base.

NumPy/SciPy calculam somente referências de validação. Os resultados
analíticos e os gráficos publicados usam exclusivamente minhastats.
"""

from __future__ import annotations

import csv
import json
import math
import platform
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

import minhastats as ms
from analise import descobertas, graficos_descobertas, textos_descobertas
from dados import CATEGORICAS, NUMERICAS, carregar_dados

RTOL, ATOL = 1e-9, 1e-10


def _representar(valores):
    vetor = np.asarray(valores, dtype=float).reshape(-1)
    if len(vetor) == 1:
        return format(float(vetor[0]), ".15g")
    if len(vetor) <= 20:
        return json.dumps(vetor.tolist(), ensure_ascii=False)
    return (f"vetor com {len(vetor)} componentes; primeiros="
            f"{json.dumps(vetor[:5].tolist())}; último={float(vetor[-1]):.15g}")


def validar(quadro):
    """Compare todo componente numérico, registrando evidência por função."""
    linhas = []

    def adicionar(funcao, cenario, proprio, referencia, biblioteca):
        a = np.asarray(proprio, dtype=float).reshape(-1)
        b = np.asarray(referencia, dtype=float).reshape(-1)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f"Referência incompatível ou não finita: {funcao}, {cenario}")
        diferenca = np.abs(a - b)
        limites = ATOL + RTOL * np.abs(b)
        passou = bool(np.all(diferenca <= limites))
        linhas.append({
            "funcao": funcao, "dados": cenario,
            "valor_proprio": _representar(a), "valor_referencia": _representar(b),
            "referencia": biblioteca, "componentes_comparados": len(a),
            "diferenca_absoluta_max": float(diferenca.max(initial=0)),
            "limite_tolerancia_max": float(limites.max(initial=0)),
            "rtol": RTOL, "atol": ATOL, "passou": passou,
        })
        if not passou:
            raise AssertionError(f"Divergência acima da tolerância: {funcao}, {cenario}")

    for coluna in NUMERICAS:
        x = quadro[coluna].tolist()
        a = np.asarray(x, dtype=float)
        adicionar("media", coluna, ms.media(x), np.mean(a), "numpy.mean")
        adicionar("mediana", coluna, ms.mediana(x), np.median(a), "numpy.median")
        valores, contagens = np.unique(a, return_counts=True)
        adicionar("moda", coluna, ms.moda(x), valores[contagens == contagens.max()], "numpy.unique + frequência máxima; todas as modas")
        adicionar("amplitude", coluna, ms.amplitude(x), np.ptp(a), "numpy.ptp")
        adicionar("quartis", coluna, ms.quartis(x), np.percentile(a, [25, 50, 75], method="linear"), "numpy.percentile(method=linear)")
        for p in [0, 10, 25, 50, 75, 90, 100]:
            adicionar(f"percentil_{p}", coluna, ms.percentil(x, p), np.percentile(a, p, method="linear"), "numpy.percentile(method=linear)")
        for amostral, tipo in [(True, "amostral"), (False, "populacional")]:
            ddof = int(amostral)
            adicionar(f"variancia_{tipo}", coluna, ms.variancia(x, amostral), np.var(a, ddof=ddof), f"numpy.var(ddof={ddof})")
            adicionar(f"desvio_padrao_{tipo}", coluna, ms.desvio_padrao(x, amostral), np.std(a, ddof=ddof), f"numpy.std(ddof={ddof})")
            adicionar(f"coeficiente_variacao_{tipo}", coluna, ms.coeficiente_variacao(x, amostral), abs(stats.variation(a, ddof=ddof)) * 100, f"abs(scipy.stats.variation(ddof={ddof}))*100")

        resumo = ms.resumo(x)
        chaves = ["n", "minimo", "maximo", "media", "mediana", "amplitude", "q1", "q2", "q3",
                  "variancia_amostral", "variancia_populacional", "desvio_padrao_amostral",
                  "desvio_padrao_populacional", "coeficiente_variacao"]
        esperado = [len(a), np.min(a), np.max(a), np.mean(a), np.median(a), np.ptp(a),
                    *np.percentile(a, [25, 50, 75], method="linear"), np.var(a, ddof=1), np.var(a),
                    np.std(a, ddof=1), np.std(a), abs(stats.variation(a, ddof=1))*100]
        adicionar("resumo", coluna, [resumo[c] for c in chaves], esperado, "NumPy/SciPy: n, mínimo, máximo, média, mediana, amplitude, Q1/Q2/Q3, variâncias, desvios, CV")
        adicionar("resumo_moda", coluna, resumo["moda"], valores[contagens == contagens.max()], "numpy.unique + frequência máxima")

        tabela = ms.frequencias(x)
        nclasses = math.ceil(1 + math.log2(len(x)))
        freq, bordas = np.histogram(a, bins=nclasses)
        adicionar("frequencias_bordas", coluna, [r["inferior"] for r in tabela] + [tabela[-1]["superior"]], bordas, "numpy.histogram; Sturges")
        adicionar("frequencias_contagens", coluna, [r["frequencia"] for r in tabela], freq, "numpy.histogram")
        adicionar("frequencias_relativas", coluna, [r["relativa"] for r in tabela], freq / len(a), "numpy.histogram / n")
        adicionar("frequencias_acumuladas", coluna, [r["acumulada"] for r in tabela], np.cumsum(freq), "numpy.cumsum(histograma)")

        out = ms.outliers_iqr(x)
        q1, q3 = np.percentile(a, [25, 75], method="linear")
        iqr = stats.iqr(a, interpolation="linear")
        inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mascara = (a < inferior) | (a > superior)
        campos = ["q1", "q3", "iqr", "limite_inferior", "limite_superior", "bigode_inferior", "bigode_superior"]
        adicionar("outliers_iqr_limites", coluna, [out[c] for c in campos],
                  [q1, q3, iqr, inferior, superior, np.min(a[~mascara]), np.max(a[~mascara])], "numpy.percentile + scipy.stats.iqr + extremos internos")
        adicionar("outliers_iqr_valores", coluna, out["outliers"], a[mascara], "máscara NumPy fora dos limites IQR")

    for nome, x in [("fixture unimodal [1,2,2,3]", [1, 2, 2, 3]),
                    ("fixture empate [1,1,2,2]", [1, 1, 2, 2])]:
        valores, contagens = np.unique(x, return_counts=True)
        adicionar("moda_fixture", nome, ms.moda(x), valores[contagens == contagens.max()], "numpy.unique + frequência máxima; todas as modas")
    adicionar("moda_unimodal_scipy", "fixture unimodal [1,2,2,3]", ms.moda([1, 2, 2, 3]),
              [float(stats.mode([1, 2, 2, 3], keepdims=False).mode)], "scipy.stats.mode; somente caso unimodal")

    x, y = quadro["temp"].tolist(), quadro["cnt"].tolist()
    for amostral, tipo in [(True, "amostral"), (False, "populacional")]:
        adicionar(f"covariancia_{tipo}", "temp × cnt", ms.covariancia(x, y, amostral), np.cov(x, y, ddof=int(amostral))[0, 1], f"numpy.cov(ddof={int(amostral)})")
    adicionar("correlacao", "temp × cnt", ms.correlacao(x, y), stats.pearsonr(x, y).statistic, "scipy.stats.pearsonr")
    adicionar("correlacao_numpy", "temp × cnt", ms.correlacao(x, y), np.corrcoef(x, y)[0, 1], "numpy.corrcoef")
    b0, b1, r2 = ms.regressao_linear(x, y)
    reta = stats.linregress(x, y)
    for nome, proprio, referencia in [("intercepto", b0, reta.intercept), ("inclinacao", b1, reta.slope), ("r2", r2, reta.rvalue**2)]:
        adicionar(f"regressao_{nome}", "temp × cnt", proprio, referencia, "scipy.stats.linregress")

    for coluna in CATEGORICAS:
        categorias = quadro[coluna].tolist()
        tabela = ms.frequencias_categoricas(categorias)
        valores, contagens = np.unique(categorias, return_counts=True)
        mapa = dict(zip(valores.tolist(), contagens.tolist()))
        referencia = np.asarray([mapa[r["categoria"]] for r in tabela])
        adicionar("frequencias_categoricas", coluna, [r["frequencia"] for r in tabela], referencia, "numpy.unique; ordem do núcleo preservada")
        adicionar("frequencias_categoricas_relativas", coluna, [r["relativa"] for r in tabela], referencia/len(categorias), "numpy.unique / n")
        adicionar("frequencias_categoricas_acumuladas", coluna, [r["acumulada"] for r in tabela], np.cumsum(referencia), "numpy.cumsum")

    mu, sd = ms.media(x), ms.desvio_padrao(x, False)
    grade = [-1, 0, .1, .5, 1, 2]
    adicionar("normal_pdf", "temp: grade [-1,0,0.1,0.5,1,2]", [ms.normal_pdf(v, mu, sd) for v in grade], stats.norm.pdf(grade, loc=mu, scale=sd), "scipy.stats.norm.pdf")
    adicionar("uniforme_pdf", "temp: grade [-1,0,0.1,0.5,1,2]", [ms.uniforme_pdf(v, min(x), max(x)) for v in grade], stats.uniform.pdf(grade, loc=min(x), scale=max(x)-min(x)), "scipy.stats.uniform.pdf")
    adicionar("exponencial_pdf", "temp: grade [-1,0,0.1,0.5,1,2]", [ms.exponencial_pdf(v, 1/mu) for v in grade], stats.expon.pdf(grade, scale=mu), "scipy.stats.expon.pdf")

    rng = random.Random(42)
    caras = np.asarray([int(rng.random() < .5) for _ in range(5000)])
    adicionar("simular_lgn", "moeda justa; 5.000 lançamentos; semente 42", ms.simular_lgn(5000, seed=42), np.cumsum(caras)/np.arange(1, 5001), "mesmos sorteios Python; frequência acumulada NumPy")
    casual = quadro["casual"].tolist()
    rng = random.Random(42)
    sorteios = np.asarray([[rng.choice(casual) for _ in range(30)] for _ in range(1000)])
    adicionar("simular_tcl", "casual; n=30; 1.000 repetições; semente 42", ms.simular_tcl(casual, 30, 1000, seed=42), np.mean(sorteios, axis=1), "mesmos sorteios Python; média por linha NumPy")
    return linhas


FORMULAS = r"""
### Tendência central e posição

Para as observações $x_1,\ldots,x_n$, seja $x_{(1)}\leq\cdots\leq x_{(n)}$ a sequência ordenada.

$$\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad A=x_{(n)}-x_{(1)}.$$

A mediana é $x_{((n+1)/2)}$ quando $n$ é ímpar, e $(x_{(n/2)}+x_{(n/2+1)})/2$ quando $n$ é par.
A moda é o conjunto $M=\{v:f(v)=\max_u f(u)\}$, devolvido como lista ordenada. Empates são preservados; se todos os valores forem distintos, todos integram o empate de frequência 1, sem declarar uma moda única.

Percentis usam o método linear (tipo 7). Com índices iniciando em zero, $h=(n-1)p/100$, $j=\lfloor h\rfloor$ e $g=h-j$:

$$P_p=(1-g)x_{[j]}+g x_{[\lceil h\rceil]},\qquad Q_1=P_{25},\quad Q_2=P_{50},\quad Q_3=P_{75}.$$

Essa convenção foi escolhida para compatibilidade explícita com `numpy.percentile(method="linear")`; outros métodos podem produzir quartis diferentes. [Documentação NumPy](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html).

### Dispersão, covariância e correlação

$$s^2=\frac{\sum_i(x_i-\bar{x})^2}{n-1},\qquad \sigma^2=\frac{\sum_i(x_i-\bar{x})^2}{n},\qquad s=\sqrt{s^2},\quad \sigma=\sqrt{\sigma^2}.$$

O modo amostral exige $n\geq2$; o populacional descreve a base empírica completa. Isso não significa que a base seja um censo de todos os contextos de mobilidade.

$$CV_{\text{amostral}}=100\frac{s}{|\bar{x}|},\qquad CV_{\text{populacional}}=100\frac{\sigma}{|\bar{x}|}.$$

O módulo da média é uma convenção declarada. O CV é considerado indefinido se $|\bar{x}|\leq10^{-12}\max(1,\max_i|x_i|)$; não se publica infinito ou um zero artificial. Para escalas normalizadas, o CV depende da origem da escala e deve ser interpretado com cuidado.

$$\operatorname{Cov}_{s}(X,Y)=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{n-1},\qquad \operatorname{Cov}_{p}(X,Y)=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{n}.$$

$$r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}.$$

Vetores precisam ter o mesmo tamanho; a correlação exige pelo menos dois pares e variáveis não constantes. A implementação centraliza os dados e normaliza os desvios antes dos produtos, reduzindo problemas de escala numérica.

### Frequências, densidade e valores atípicos

O número padrão de classes é $k=\lceil1+\log_2n\rceil$ (Sturges), com largura $w=(\max x-\min x)/k$. Os intervalos são fechados à esquerda e abertos à direita; o último inclui o máximo. Uma base constante recebe um intervalo representável ao redor do valor.

$$f_j=\sum_i\mathbf{1}(x_i\in C_j),\qquad p_j=f_j/n,\qquad F_j=\sum_{\ell\leq j}f_\ell.$$

Para sobrepor uma densidade teórica, a altura de cada barra é $h_j=p_j/w_j$, de modo que $\sum_j h_jw_j=1$. Um histograma de contagens não pode ser comparado diretamente com uma PDF sem esse ajuste. Categorias usam contagens por rótulo; a ordem de apresentação não cria uma escala ordinal.

$$IQR=Q_3-Q_1,\qquad L=Q_1-1{,}5IQR,\qquad U=Q_3+1{,}5IQR.$$

Valores menores que $L$ ou maiores que $U$ são sinalizados. Os bigodes do boxplot ficam nos extremos efetivamente observados dentro desses limites, e não nos próprios limites teóricos. A classificação não comprova erro de medição. A interpretação textual utiliza $(\bar{x}-\operatorname{mediana})/\sigma$ e limiar de proximidade $0{,}1$ como heurística; não é um teste de simetria ou normalidade.

### Regressão linear por mínimos quadrados

O núcleo escolhe $b_0,b_1$ para minimizar $\sum_i(y_i-b_0-b_1x_i)^2$:

$$b_1=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2},\qquad b_0=\bar{y}-b_1\bar{x},\qquad \widehat{y}=b_0+b_1x.$$

$$R^2=1-\frac{\sum_i(y_i-\widehat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}.$$

O cálculo dos resíduos é feito na forma centrada equivalente. $X$ e $Y$ constantes são rejeitados, porque a inclinação ou o denominador de $R^2$ seriam indefinidos. Na regressão simples com intercepto, $R^2=r^2$, identidade usada na referência de validação. A comparação externa utiliza [SciPy linregress](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).

### Densidades teóricas

$$f_N(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right],\quad x\in\mathbb{R},\quad\sigma>0.$$

$$f_U(x)=\begin{cases}\frac{1}{b-a},&a\leq x\leq b\\0,&\text{caso contrário},\end{cases}\qquad a<b.$$

$$f_E(x)=\begin{cases}\lambda e^{-\lambda x},&x\geq0\\0,&x<0,\end{cases}\qquad\lambda>0.$$

A Normal usa $\widehat\mu=\bar{x}$ e $\widehat\sigma=\sqrt{\sum_i(x_i-\bar{x})^2/n}$. A Uniforme usa $a=\min x$, $b=\max x$. A Exponencial sem deslocamento usa $\widehat\lambda=1/\bar{x}$ e só é oferecida para observações não negativas com média positiva. Esses ajustes simples permitem comparação exploratória, sem declarar que o modelo seja verdadeiro.

### Lei dos Grandes Números e Teorema Central do Limite

Na experiência da moeda, $X_i\sim\operatorname{Bernoulli}(0{,}5)$ independentes e $S_n=\sum_iX_i$:

$$\frac{S_n}{n}\xrightarrow[]{P}0{,}5.$$

A LGN descreve aproximação com o crescimento de $n$, não aproximação monotônica em toda trajetória finita. No TCL, sorteios independentes com reposição da distribuição empírica produzem médias $\bar{X}_n$:

$$\frac{\sqrt n(\bar{X}_n-\mu)}{\sigma}\xrightarrow[]{d}\mathcal N(0,1),\qquad \operatorname{DP}(\bar{X}_n)=\frac{\sigma}{\sqrt n}.$$

Cada repetição usa uma nova amostra. A média empírica finita e sua variância fornecem os parâmetros de referência; aumentar o número de repetições melhora a aproximação do histograma simulado, enquanto aumentar o tamanho de cada amostra altera a distribuição das médias. São controles com papéis diferentes.
"""


def escrever_relatorio(quadro, resultados, linhas):
    identidade = json.loads((ROOT / "entrega.json").read_text(encoding="utf-8"))
    manifesto = json.loads((ROOT / "dados/metadados.json").read_text(encoding="utf-8"))
    escolhido = [l for l in linhas if (l["dados"] == "cnt" and l["funcao"] in {
        "media", "mediana", "moda", "amplitude", "quartis", "percentil_10", "percentil_90",
        "variancia_amostral", "variancia_populacional", "desvio_padrao_amostral", "desvio_padrao_populacional",
        "coeficiente_variacao_amostral", "coeficiente_variacao_populacional"}) or l["dados"] == "temp × cnt" or l["funcao"] == "moda_unimodal_scipy"]
    tabela = ["| Função / cenário | Próprio | Referência | Diferença absoluta |",
              "|---|---:|---:|---:|"]
    for linha in escolhido:
        tabela.append(f"| {linha['funcao']} ({linha['dados']}) | {linha['valor_proprio']} | {linha['valor_referencia']} | {linha['diferenca_absoluta_max']:.3e} |")
    max_erro = max(l["diferenca_absoluta_max"] for l in linhas)
    componentes = sum(l["componentes_comparados"] for l in linhas)
    registros_pt = f"{len(quadro):,}".replace(",", ".")
    componentes_pt = f"{componentes:,}".replace(",", ".")
    t = resultados["temperatura"]
    y = quadro["cnt"].tolist()
    resumo = ms.resumo(y)
    casual = quadro["casual"].tolist()
    f = ms.simular_lgn(5000, seed=42)
    simuladas = ms.simular_tcl(casual, 30, 1000, seed=42)
    nome = identidade.get("nome", "PREENCHER")
    matricula = identidade.get("matricula", "PREENCHER")
    texto = f"""# Pedal em Dados — Relatório de sistematização

**Disciplina:** Matemática e Estatística para Computação  
**Modalidade:** projeto individual  
**Estudante:** {nome}  
**Matrícula:** {matricula}  
**Fonte dos campos de identificação:** `entrega.json`.

Se houver `PREENCHER`, o campo ainda precisa ser completado pelo estudante antes da entrega. Este relatório não atribui nomes ou matrículas presumidos.

## 1. Objetivo e organização

O projeto implementa um laboratório estatístico interativo para investigar a relação entre o contexto de uma hora e a demanda de bicicletas compartilhadas. A aplicação integra exploração descritiva, experimentos de probabilidade, comparação de distribuições e regressão linear, com um núcleo próprio verificável. O desenvolvimento e a apresentação são individuais.

O núcleo `minhastats.py` emprega Python padrão; não importa NumPy, Pandas, SciPy ou `statistics`. `dados.py` carrega e valida os registros; `graficos.py` desenha valores previamente calculados pelo núcleo; `analise.py` reúne descobertas; `app.py` oferece os sete módulos Streamlit. NumPy e SciPy são referências independentes na validação, e Pandas é utilizado para carregar, selecionar e organizar registros. Os números das descobertas abaixo são calculados por `minhastats`, incluindo frequências que dão origem às figuras.

## 2. Dataset, escolha e tratamento

O **Bike Sharing**, de **Hadi Fanaee-T (2013)**, contém contagens do Capital Bikeshare em Washington, D.C., associadas a meteorologia e calendário em 2011–2012. Foi escolhida a versão horária `hour.csv`, que permite comparar tipos de dia e horários de alta demanda. A fonte é o [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), com licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

A cópia analisada possui **{registros_pt} registros e 17 colunas originais**, atendendo ao mínimo de 1.000 registros. O catálogo informa 17.389 instâncias, enquanto o CSV e seu README identificam 17.379; a análise utiliza o arquivo efetivo. Há **{len(NUMERICAS)} variáveis numéricas** nos seletores e **{len(CATEGORICAS)} categóricas**, derivadas dos códigos originais.

As variáveis numéricas são temperatura, sensação térmica, umidade e vento normalizados, além das contagens `casual`, `registered` e `cnt`. As categóricas são clima, tipo de dia, ano e feriado. O identificador, a data e códigos numéricos de categorias ficam fora do seletor contínuo. `cnt = casual + registered`; uma associação entre parte e total é estrutural e não constitui evidência causal.

O CSV foi preservado integralmente; a leitura apenas acrescenta rótulos e interpreta a data. Nenhum registro foi excluído ou imputado. Horas ausentes não foram transformadas em zeros, e valores sinalizados pelo IQR foram mantidos. A validação rejeita registros inconsistentes, em vez de corrigir silenciosamente a base. O conjunto de verificações inclui esquema, número de linhas, nulos, finitude, domínios, contagens inteiras não negativas, soma dos tipos de usuário, unicidade e coerência de datas.

Há divergências entre o catálogo e o README sobre a normalização da temperatura e os nomes associados às estações. Por isso, as medidas meteorológicas permanecem na escala normalizada original; não se atribuem °C aos seus valores. O código `season` é preservado, mas não traduzido nem usado nos seletores. O dicionário, as decisões e a atribuição completa estão em [dados/FONTE.md](dados/FONTE.md).

**Integridade da cópia:** SHA-256 do CSV `{manifesto['sha256_csv']}`. O manifesto [dados/metadados.json](dados/metadados.json) registra origem, momento de obtenção e hashes. Um hash permite verificar igualdade de arquivos; isoladamente não comprova sua procedência.

## 3. Núcleo estatístico e decisões matemáticas

As funções validam entradas vazias, valores não numéricos/não finitos e parâmetros inválidos. Medidas indefinidas produzem erro explícito; no resumo, valores amostrais/CV indefinidos são apresentados como indisponíveis. Somas usam `math.fsum` para reduzir erro de arredondamento, sem substituir as fórmulas por chamadas estatísticas prontas.
"""
    texto += FORMULAS
    texto += fr"""

## 4. Validação reproduzível

O comando `python scripts/gerar_resultados.py` calculou **{len(linhas)} comparações aprovadas**, envolvendo **{componentes_pt} componentes numéricos**. A comparação é componente a componente, segundo

$$|v_{{\mathrm{{próprio}}}}-v_{{\mathrm{{ref}}}}|\leq10^{{-10}}+10^{{-9}}|v_{{\mathrm{{ref}}}}|.$$

A maior diferença absoluta observada entre todos os componentes foi **{max_erro:.6e}**. Esse máximo considera medidas em escalas diferentes; o aceite sempre usa a tolerância aplicada ao valor de referência de cada componente. A evidência completa, com função, dados, referência, valores, erro e tolerância, está em [docs/validacao.csv](docs/validacao.csv). Para vetores longos, o CSV resume a exibição, mas compara todos os componentes e registra sua quantidade.

As sete variáveis reais são verificadas para média, mediana, todas as modas, amplitude, percentis, quartis, variâncias, desvios, CV, resumo, frequências e IQR. O par temperatura–demanda verifica covariâncias, Pearson e regressão. Também se verificam frequências categóricas, densidades e trajetórias simuladas. Na validação das simulações, a referência utiliza exatamente os mesmos sorteios e calcula frequências/médias com NumPy; isso verifica o cálculo, sem ser uma prova empírica dos teoremas.

**Convenção da moda:** `scipy.stats.mode` devolve uma única moda; por isso sua comparação direta é feita no caso unimodal `[1,2,2,3]`. Empates, como `[1,1,2,2]`, são comparados ao conjunto de valores de frequência máxima obtido com `numpy.unique`. [Documentação SciPy mode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html).

Tabela resumida da execução atual (valores com até 15 algarismos significativos):

{chr(10).join(tabela)}

Ambiente efetivamente usado na geração: Python {platform.python_version()}, NumPy {np.__version__} e SciPy {scipy.__version__}. Além dessas comparações, os testes automatizados em `test_minhastats.py` cobrem amostras constantes, unitárias, vazias, assimetria, multimodalidade, média próxima de zero, deslocamentos grandes, vetores incompatíveis, bordas de intervalos, sementes e parâmetros inválidos. `test_dados.py` verifica a base real e arquivos deliberadamente corrompidos. Para executar toda a suíte: `python -m pytest -q`. O número de comparações deste relatório não é apresentado como número de casos pytest.

## 5. Módulos e evidências da interface

### Módulo 0 — Dados reais

A tela inicial apresenta dimensão da base, período, amostra de registros, dicionário e acesso ao CSV. O usuário pode verificar que as unidades observacionais são horas. A fonte original está acessível na barra lateral. Os totais exibidos correspondem ao CSV efetivamente carregado.

![Módulo 0: apresentação e dicionário do dataset](docs/modulo_0.png)

### Módulo 1 — Núcleo estatístico próprio

O seletor permite calcular todas as medidas de uma variável, consultar a implementação e abrir a tabela de validação. Em `cnt`, a média é {resumo['media']:.6f}, a mediana é {resumo['mediana']:.0f} e o desvio amostral é {resumo['desvio_padrao_amostral']:.6f} aluguéis por hora. As convenções amostral/populacional e de percentis são apresentadas explicitamente.

![Módulo 1: medidas calculadas pelo núcleo próprio](docs/modulo_1.png)

### Módulo 2 — Estatística descritiva interativa

Para variáveis numéricas, o módulo apresenta tendência central, dispersão, tabela de classes, histograma, boxplot e candidatos a outliers por IQR. Para categorias, apresenta barras e tabela de frequências. A interpretação automática informa o caráter heurístico da sugestão de assimetria. O agrupamento em classes também é aplicado às contagens discretas para facilitar a visualização.

![Módulo 2: distribuição, tabela e valores atípicos](docs/modulo_2.png)

### Módulo 3 — Probabilidade e Monte Carlo

A LGN usa lançamentos de moeda justa; com 5.000 lançamentos e semente 42, a frequência final de caras é **{f[-1]:.4f}**. O TCL reamostra a variável escolhida com reposição. Para `casual`, tamanho 30, 1.000 repetições e semente 42, a média das médias é **{ms.media(simuladas):.6f}**, contra **{ms.media(casual):.6f}** na base; o desvio das médias é **{ms.desvio_padrao(simuladas):.6f}**, e a referência $\sigma/\sqrt n$ é **{ms.desvio_padrao(casual, False)/math.sqrt(30):.6f}**. Diferenças são esperadas em uma simulação finita.

O usuário controla semente, número de lançamentos, variável, tamanho amostral e repetições. Comparar tamanhos 2, 30 e 100 evidencia a mudança da forma e da dispersão das médias. A independência é construída no sorteio com reposição; ela não é pressuposta para horas consecutivas da base.

![Módulo 3: experiências LGN e TCL com controles](docs/modulo_3.png)

![Módulo 3: amostragem e distribuição das médias no TCL](docs/modulo_3_tcl.png)

### Módulo 4 — Distribuições teóricas

A Normal está sempre sobreposta ao histograma de densidade; o usuário escolhe Uniforme ou Exponencial como segunda candidata, conforme o suporte da variável. O histograma tem área total unitária, possibilitando comparação na mesma escala das PDFs. Parâmetros são estimados pelo núcleo próprio a partir da variável escolhida.

A Normal é simétrica e tem suporte ilimitado: pode atribuir massa a valores meteorológicos fora de [0,1] ou a contagens negativas. A Uniforme tem densidade constante entre extremos observados, sendo inadequada visualmente quando há picos ou concentração central; o intervalo estimado não estabelece um limite físico. A Exponencial tem máximo em zero e queda monotônica, sendo incapaz de representar bem um pico distante de zero. Também não respeita um teto conhecido da codificação meteorológica. Compare picos, dispersão e caudas, inclusive a massa fora da faixa plotada; o gráfico mostra a faixa observada e não toda a massa das distribuições.

`casual`, `registered` e `cnt` são contagens: as três PDFs contínuas servem somente como aproximações visuais, e a altura de uma PDF não é a probabilidade de uma contagem exata. O laboratório não conclui aderência por teste formal nem seleciona um modelo apenas pela aparência. Nas variáveis limitadas e nas distribuições com assimetria, uma curva próxima em parte da faixa pode continuar sendo um modelo global ruim.

![Módulo 4: histograma normalizado e candidatas teóricas](docs/modulo_4.png)

### Módulo 5 — Correlação e regressão linear

Dois seletores definem X e Y; o módulo mostra dispersão, Pearson, reta, equação e $R^2$. O campo de entrada calcula uma predição para X, com avisos de extrapolação ou de contagem negativa. Para temperatura normalizada e total, a equação ajustada é $\widehat{{cnt}}={t['b0']:.6f}+{t['b1']:.6f}\,temp$. Uma variação de 0,1 na temperatura normalizada corresponde a **{t['b1']*.1:.4f}** aluguéis por hora na reta; esse valor não pode ser interpretado como efeito de 0,1 °C.

O intercepto é o valor extrapolado da reta em X=0; sua utilidade física depende da faixa observada. A inclinação expressa associação média no ajuste, e $R^2$ é calculado na própria base. Não houve separação de treino/teste ou validação temporal de previsões.

![Módulo 5: correlação, reta e predição interativa](docs/modulo_5.png)

### Módulo 6 — Relatório de descobertas

O módulo reúne três resultados descritivos sustentados por números e gráficos gerados pela aplicação, os mesmos reproduzidos na seção seguinte. O usuário pode baixar as evidências em JSON e este relatório. As afirmações mantêm as limitações de generalização e causalidade.

![Módulo 6: descobertas reproduzidas pela aplicação](docs/modulo_6.png)

## 6. Três descobertas sustentadas pela base
"""
    for numero, (titulo, paragrafo) in enumerate(textos_descobertas(resultados), 1):
        texto += f"\n### Descoberta {numero} — {titulo}\n\n{paragrafo}\n\n![Figura {numero}: {titulo}](docs/figura_{numero}.png)\n"
        if numero == 1:
            grupos = resultados["grupos"]
            texto += (f"\nAs médias correspondentes são {grupos[0]['media']:.4f} e {grupos[1]['media']:.4f}, "
                      "respectivamente. A figura usa medianas, que são menos sensíveis a horas com contagens muito altas. "
                      "São comparações por hora observada, não por pessoa ou por quantidade de dias. Os tamanhos diferentes "
                      "dos grupos e a mistura de horários devem ser considerados antes de generalizar.\n")
        elif numero == 2:
            texto += (f"\nA reta explica aproximadamente {t['r2']*100:.2f}% da variação de `cnt` nesta base. "
                      "Horário, ano, tipo de dia e outras condições podem estar associados tanto à temperatura quanto à demanda. "
                      "Não se estima o efeito causal de aquecer o ambiente, e o ajuste não demonstra capacidade de prever um período futuro.\n")
        else:
            texto += ("\nA figura conta as horas acima do limite superior por hora do relógio, juntando todos os tipos de dia. "
                      "O percentual de pico informado no texto usa um filtro adicional: somente dias úteis às 17h ou 18h, "
                      "com denominador igual ao total de horas sinalizadas. Portanto, a figura e esse percentual têm recortes explícitos diferentes. "
                      "Um único limite global ignora padrões por horário; em uma análise posterior seria útil comparar referências específicas por faixa horária.\n")
    texto += """

## 7. Limitações e possibilidades de continuidade

O estudo descreve um sistema específico em 2011–2012. Não se presume representatividade de outras cidades, pessoas ou da demanda atual. A unidade observacional é uma hora registrada; há dependência temporal, sazonalidade e horários ausentes. Isso impede tratar automaticamente os registros como uma amostra aleatória independente para inferência. As comparações de grupos são exploratórias e não controlam confundidores.

A regra IQR identifica extremos em relação à distribuição global, podendo sinalizar picos operacionais legítimos. As PDFs escolhidas têm limitações de suporte e forma. A regressão é simples, pode produzir previsões fisicamente impossíveis e usa todos os dados no ajuste; seu R² é descritivo, sem estimativa de desempenho futuro. Correlação não implica causalidade.

Uma continuidade possível seria incorporar hora, ano e calendário em modelos adequados a contagens e avaliá-los com divisão temporal entre ajuste e teste. Essa extensão precisaria preservar a separação entre dados usados para escolher o modelo e dados usados para avaliar previsões; ela não foi implementada neste laboratório.

## 8. Reprodução e entrega individual

Na pasta do projeto, após instalar as dependências de `requirements.txt`:

```text
python -m pytest -q
python scripts/gerar_resultados.py
python -m streamlit run app.py
```

O segundo comando recria este relatório, a validação CSV, as descobertas JSON e as três figuras a partir da base. As capturas `docs/modulo_0.png` a `docs/modulo_6.png` documentam a interface e são produzidas ao operar a aplicação. O download dos dados é separado e explícito: `python scripts/baixar_dados.py`; a cópia distribuída permite executar a aplicação sem baixar a base novamente.

A identificação e os links finais devem estar preenchidos em `entrega.json`; não se deve enviar a versão com `PREENCHER`. O estudante deve revisar os resultados, compreender as fórmulas, apresentar pessoalmente o funcionamento no vídeo e manter um histórico verdadeiro de alterações no repositório. A geração deste material não substitui a demonstração nem comprova autoria individual de cada etapa.

## Referências

- Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. DOI: [10.24432/C5W894](https://doi.org/10.24432/C5W894). Documentação original preservada em `dados/Readme_original.txt`.
- NumPy Developers. [numpy.percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html). Convenção linear e métodos de percentis usados na referência.
- SciPy Developers. [scipy.stats.mode](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html). Convenção de retorno de uma moda.
- SciPy Developers. [scipy.stats.linregress](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html). Ajuste de mínimos quadrados usado na validação.
- Implementações e evidências reproduzíveis deste projeto: `minhastats.py`, `test_minhastats.py`, `test_dados.py`, `scripts/gerar_resultados.py`, `docs/validacao.csv` e `docs/descobertas.json`.
"""
    (ROOT / "RELATORIO.md").write_text(texto, encoding="utf-8")


def main():
    quadro = carregar_dados()
    linhas = validar(quadro)
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    with (docs / "validacao.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    resultados = descobertas(quadro)
    (docs / "descobertas.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for numero, figura in enumerate(graficos_descobertas(quadro, resultados), 1):
        figura.savefig(docs / f"figura_{numero}.png", dpi=180, facecolor="white")
        plt.close(figura)
    escrever_relatorio(quadro, resultados, linhas)
    print(f"{len(linhas)} comparações aprovadas; relatório, CSV, JSON e 3 figuras atualizados.")


if __name__ == "__main__":
    main()
