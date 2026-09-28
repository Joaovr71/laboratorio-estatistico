"""Gráficos usam medidas de minhastats; Matplotlib apenas desenha."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import minhastats as ms

VERDE = "#147D73"
LARANJA = "#D87937"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#253B4A", "font.size": 10,
                     "axes.titlesize": 13, "figure.dpi": 120})

def base(titulo, xlabel, ylabel, tamanho=(9, 4)):
    fig, ax = plt.subplots(figsize=tamanho, layout="constrained")
    ax.set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    return fig, ax

def histograma(dados, rotulo, densidade=False, ax=None):
    if ax is None:
        fig, ax = base("Distribuição observada", rotulo, "Densidade" if densidade else "Frequência")
    else:
        fig = ax.figure
    tabela = ms.frequencias(dados)
    larguras = [r["superior"] - r["inferior"] for r in tabela]
    alturas = [r["relativa"] / w if densidade else r["frequencia"] for r, w in zip(tabela, larguras)]
    ax.bar([r["inferior"] for r in tabela], alturas, width=larguras, align="edge",
           color=VERDE, edgecolor="white", alpha=.75, label="Dados observados")
    ax.set(xlabel=rotulo, ylabel="Densidade" if densidade else "Frequência")
    return fig, ax

def boxplot(dados, rotulo):
    o = ms.outliers_iqr(dados)
    fig, ax = base("Dispersão e valores atípicos (1,5 × IQR)", rotulo, "", (9, 2.8))
    ax.bxp([dict(med=ms.mediana(dados), q1=o["q1"], q3=o["q3"],
                 whislo=o["bigode_inferior"], whishi=o["bigode_superior"],
                 fliers=o["outliers"])], orientation="horizontal", patch_artist=True,
           boxprops={"facecolor": "#BFE2DB"}, flierprops={"marker": ".", "alpha": .25})
    ax.set_yticks([])
    return fig

def categorias(dados, rotulo):
    tabela = ms.frequencias_categoricas(dados)
    fig, ax = base("Frequências por categoria", rotulo, "Registros")
    ax.bar([str(r["categoria"]) for r in tabela], [r["frequencia"] for r in tabela], color=VERDE)
    ax.tick_params(axis="x", rotation=15)
    return fig

def lgn(frequencias):
    fig, ax = base("Moeda justa: frequência acumulada de caras", "Lançamentos", "Frequência relativa")
    ax.plot(range(1, len(frequencias)+1), frequencias, color=VERDE, linewidth=1)
    ax.axhline(.5, color=LARANJA, linestyle="--", label="Probabilidade teórica = 0,5")
    ax.set_xscale("log")
    ax.legend()
    return fig

def tcl(dados, medias, tamanho, rotulo):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    histograma(dados, rotulo, True, axes[0])
    axes[0].set_title("População empírica original")
    histograma(medias, "Médias amostrais", True, axes[1])
    mu, se = ms.media(dados), ms.desvio_padrao(dados, False) / tamanho**.5
    lo, hi = min(min(medias), mu-4*se), max(max(medias), mu+4*se)
    grid = [lo+(hi-lo)*i/300 for i in range(301)]
    axes[1].plot(grid, [ms.normal_pdf(v, mu, se) for v in grid], color=LARANJA,
                 label="Normal: μ e σ/√n da base")
    axes[1].set_title(f"Médias de amostras com n = {tamanho}")
    axes[1].legend(fontsize=8)
    return fig

def distribuicoes(dados, rotulo, modelo):
    fig, ax = histograma(dados, rotulo, True)
    mu, sd = ms.media(dados), ms.desvio_padrao(dados, False)
    lo, hi = min(dados), max(dados)
    grid = [lo+(hi-lo)*i/400 for i in range(401)]
    curvas = [ms.normal_pdf(v, mu, sd) for v in grid]
    ax.plot(grid, curvas, color=LARANJA, label=f"Normal: μ={mu:.3f}, σ={sd:.3f}")
    if modelo == "Uniforme":
        curva = [ms.uniforme_pdf(v, lo, hi) for v in grid]
        texto = f"Uniforme: a={lo:.3f}, b={hi:.3f}"
    else:
        curva = [ms.exponencial_pdf(v, 1/mu) for v in grid]
        texto = f"Exponencial: λ={1/mu:.3f}"
    ax.plot(grid, curva, color="#5864A0", linestyle="--", label=texto)
    ax.legend(fontsize=8)
    ax.set_title("Dados e distribuições teóricas estimadas")
    return fig

def regressao(x, y, xlabel, ylabel):
    b0, b1, r2 = ms.regressao_linear(x, y)
    fig, ax = base("Associação e reta de mínimos quadrados", xlabel, ylabel)
    ax.scatter(x, y, s=5, alpha=.10, color=VERDE, rasterized=True)
    extremos = [min(x), max(x)]
    ax.plot(extremos, [b0+b1*v for v in extremos], color=LARANJA, linewidth=2,
            label=f"ŷ = {b0:.2f} + {b1:.2f}x | R² = {r2:.3f}")
    ax.legend(fontsize=9)
    return fig
