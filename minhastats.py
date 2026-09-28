"""Núcleo estatístico autoral do Laboratório Estatístico.

Implementações em Python padrão, sem NumPy, SciPy, pandas ou statistics.
As funções rejeitam amostras vazias, valores não numéricos e não finitos.
Percentis seguem interpolação linear no índice p(n-1)/100 (tipo 7).
"""

import math
import random
from bisect import bisect_right


def _numero(valor, nome="valor"):
    """Converte um número finito; textos e booleanos não são observações."""
    if isinstance(valor, (str, bytes, bool)):
        raise ValueError(f"{nome} deve ser numérico e finito.")
    try:
        resultado = float(valor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{nome} deve ser numérico e finito.") from exc
    if not math.isfinite(resultado):
        raise ValueError(f"{nome} deve ser numérico e finito.")
    return resultado


def _dados(dados):
    try:
        valores = [_numero(v, "Cada observação") for v in dados]
    except TypeError as exc:
        raise ValueError("Informe uma sequência de números.") from exc
    if not valores:
        raise ValueError("A amostra não pode ser vazia.")
    return valores


def _inteiro_positivo(valor, nome):
    numero = _numero(valor, nome)
    if numero < 1 or numero != int(numero):
        raise ValueError(f"{nome} deve ser inteiro positivo.")
    return int(numero)


def _media(valores):
    # fsum reduz o erro de arredondamento em somas com cancelamento.
    try:
        return math.fsum(valores) / len(valores)
    except OverflowError:
        return math.fsum(v / len(valores) for v in valores)


def _percentil_ordenado(ordenados, p):
    posicao = p * (len(ordenados) - 1) / 100.0
    inferior = math.floor(posicao)
    superior = math.ceil(posicao)
    fracao = posicao - inferior
    return (1.0 - fracao) * ordenados[inferior] + fracao * ordenados[superior]


def _pares(x, y):
    xv, yv = _dados(x), _dados(y)
    if len(xv) != len(yv):
        raise ValueError("Os vetores precisam ter o mesmo tamanho.")
    return xv, yv


def _divisor(n, amostral):
    if not isinstance(amostral, bool):
        raise ValueError("amostral deve ser True ou False.")
    if amostral and n < 2:
        raise ValueError("A medida amostral exige pelo menos 2 observações.")
    return n - 1 if amostral else n


def media(x):
    """Média aritmética: x̄ = Σxᵢ/n."""
    return _media(_dados(x))


def mediana(x):
    """Mediana: valor central, ou média dos dois centrais quando n é par."""
    return _percentil_ordenado(sorted(_dados(x)), 50.0)


def moda(x):
    """Lista ordenada de todos os valores com frequência máxima.

    Quando todos são distintos, todos são devolvidos (empate de frequência 1).
    Não confundir esta convenção com declarar uma moda única.
    """
    contagens = {}
    for valor in _dados(x):
        contagens[valor] = contagens.get(valor, 0) + 1
    maior = max(contagens.values())
    return sorted(valor for valor, frequencia in contagens.items() if frequencia == maior)


def amplitude(x):
    """Amplitude total: máximo menos mínimo."""
    valores = _dados(x)
    return max(valores) - min(valores)


def percentil(x, p):
    """Percentil p em [0,100], interpolado no índice h = p(n-1)/100."""
    percentual = _numero(p, "p")
    if not 0 <= percentual <= 100:
        raise ValueError("p deve estar entre 0 e 100.")
    return _percentil_ordenado(sorted(_dados(x)), percentual)


def quartis(x):
    """Retorna (Q1, Q2, Q3), percentis 25, 50 e 75 do tipo 7."""
    ordenados = sorted(_dados(x))
    return tuple(_percentil_ordenado(ordenados, p) for p in (25, 50, 75))


def variancia(x, amostral=True):
    """Σ(xᵢ-x̄)²/(n-1) amostral; Σ(xᵢ-x̄)²/n populacional."""
    valores = _dados(x)
    divisor = _divisor(len(valores), amostral)
    centro = _media(valores)
    return math.fsum((valor - centro) ** 2 for valor in valores) / divisor


def desvio_padrao(x, amostral=True):
    """Raiz quadrada da variância amostral ou populacional."""
    return math.sqrt(variancia(x, amostral))


def coeficiente_variacao(x, amostral=True):
    """CV (%) = 100 × desvio padrão / |média|.

    Rejeita |média| <= 1e-12 × max(1, maior |x|): valores próximos de zero
    tornam esta razão instável. A convenção com módulo evita CV negativo.
    """
    valores = _dados(x)
    centro = _media(valores)
    limite = 1e-12 * max(1.0, max(abs(v) for v in valores))
    if abs(centro) <= limite:
        raise ValueError("Coeficiente de variação indefinido: média zero ou muito próxima de zero.")
    return 100.0 * desvio_padrao(valores, amostral) / abs(centro)


def covariancia(x, y, amostral=True):
    """Σ(xᵢ-x̄)(yᵢ-ȳ)/(n-1) amostral; divisor n na populacional."""
    xv, yv = _pares(x, y)
    divisor = _divisor(len(xv), amostral)
    xm, ym = _media(xv), _media(yv)
    return math.fsum((xi - xm) * (yi - ym) for xi, yi in zip(xv, yv)) / divisor


def correlacao(x, y):
    """Pearson r = Σdxᵢdyᵢ / √(Σdxᵢ² Σdyᵢ²), em [-1,1]."""
    xv, yv = _pares(x, y)
    if len(xv) < 2:
        raise ValueError("A correlação exige pelo menos 2 pares.")
    xm, ym = _media(xv), _media(yv)
    dx, dy = [v - xm for v in xv], [v - ym for v in yv]
    sx, sy = math.sqrt(math.fsum(v * v for v in dx)), math.sqrt(math.fsum(v * v for v in dy))
    if sx == 0 or sy == 0:
        raise ValueError("A correlação é indefinida para uma variável constante.")
    # Normalizar antes dos produtos evita multiplicar duas somas grandes.
    resultado = math.fsum((a / sx) * (b / sy) for a, b in zip(dx, dy))
    return max(-1.0, min(1.0, resultado))


def regressao_linear(x, y):
    """Retorna (intercepto b0, inclinação b1, R²) por mínimos quadrados.

    b1=Σdxᵢdyᵢ/Σdxᵢ²; b0=ȳ-b1x̄; R²=1-SQres/SQtotal.
    Exige X e Y não constantes, pois R² é indefinido quando Y é constante.
    """
    xv, yv = _pares(x, y)
    if len(xv) < 2:
        raise ValueError("A regressão exige pelo menos 2 pares.")
    xm, ym = _media(xv), _media(yv)
    sxx = math.fsum((xi - xm) ** 2 for xi in xv)
    syy = math.fsum((yi - ym) ** 2 for yi in yv)
    if sxx == 0 or syy == 0:
        raise ValueError("Selecione duas variáveis não constantes para a regressão e seu R².")
    b1 = math.fsum((xi - xm) * (yi - ym) for xi, yi in zip(xv, yv)) / sxx
    b0 = ym - b1 * xm
    # Forma centrada é equivalente e reduz cancelamento para interceptos altos.
    sqres = math.fsum(((yi - ym) - b1 * (xi - xm)) ** 2 for xi, yi in zip(xv, yv))
    r2 = max(0.0, min(1.0, 1.0 - sqres / syy))
    return b0, b1, r2


def resumo(x):
    """Dicionário descritivo; medidas amostrais/CV indefinidas recebem None."""
    valores = _dados(x)
    q1, q2, q3 = quartis(valores)
    try:
        cv = coeficiente_variacao(valores)
    except ValueError:
        cv = None
    return {
        "n": len(valores), "minimo": min(valores), "maximo": max(valores),
        "media": media(valores), "mediana": q2, "moda": moda(valores),
        "amplitude": amplitude(valores), "q1": q1, "q2": q2, "q3": q3,
        "variancia_amostral": variancia(valores) if len(valores) > 1 else None,
        "variancia_populacional": variancia(valores, False),
        "desvio_padrao_amostral": desvio_padrao(valores) if len(valores) > 1 else None,
        "desvio_padrao_populacional": desvio_padrao(valores, False),
        "coeficiente_variacao": cv,
    }


def frequencias(x, classes=None):
    """Histograma com classes de mesma largura e frequência relativa em [0,1].

    Padrão: k=ceil(1+log₂n), regra de Sturges. Intervalos [inferior,superior)
    e último fechado à direita. Em amostra constante, usa [valor-0,5,valor+0,5]
    e uma classe por padrão. 'acumulada' é a contagem acumulada absoluta.
    """
    valores = _dados(x)
    menor, maior = min(valores), max(valores)
    k = _inteiro_positivo(classes, "classes") if classes is not None else math.ceil(1 + math.log2(len(valores)))
    if menor == maior:
        menor, maior = menor - 0.5, maior + 0.5
        if classes is None:
            k = 1
    largura = (maior - menor) / k
    if not math.isfinite(largura) or largura <= 0:
        raise ValueError("Não foi possível representar os intervalos nesta escala numérica.")
    bordas = [menor + i * largura for i in range(k)] + [maior]
    contagens = [0] * k
    for valor in valores:
        indice = min(k - 1, max(0, bisect_right(bordas, valor) - 1))
        contagens[indice] += 1
    acumulada, resultado = 0, []
    for indice, contagem in enumerate(contagens):
        acumulada += contagem
        resultado.append({"inferior": bordas[indice], "superior": bordas[indice + 1],
                          "frequencia": contagem, "relativa": contagem / len(valores),
                          "acumulada": acumulada})
    return resultado


def frequencias_categoricas(x):
    """Contagem de categorias, em ordem de primeira ocorrência.

    Aceita categorias textuais ou números finitos. Rejeita ausentes, listas e
    demais valores não hashable; 'acumulada' é a contagem absoluta acumulada.
    """
    try:
        valores = list(x)
    except TypeError as exc:
        raise ValueError("Informe uma sequência de categorias.") from exc
    if not valores:
        raise ValueError("A amostra não pode ser vazia.")
    contagens = {}
    for valor in valores:
        if valor is None or (not isinstance(valor, (str, bytes, bool)) and not math.isfinite(_numero(valor))):
            raise ValueError("As categorias não podem conter valores ausentes ou não finitos.")
        try:
            contagens[valor] = contagens.get(valor, 0) + 1
        except TypeError as exc:
            raise ValueError("Cada categoria precisa ser um valor simples.") from exc
    acumulada, resultado = 0, []
    for categoria, contagem in contagens.items():
        acumulada += contagem
        resultado.append({"categoria": categoria, "frequencia": contagem,
                          "relativa": contagem / len(valores), "acumulada": acumulada})
    return resultado


def outliers_iqr(x):
    """Regra de Tukey: candidatos fora de [Q1-1,5IQR, Q3+1,5IQR].

    Bigodes são os extremos efetivamente observados dentro dos limites.
    Retorna os candidatos na ordem original; não remove observações.
    """
    valores = _dados(x)
    q1, _, q3 = quartis(valores)
    iqr = q3 - q1
    inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    internos = [v for v in valores if inferior <= v <= superior]
    return {"q1": q1, "q3": q3, "iqr": iqr,
            "limite_inferior": inferior, "limite_superior": superior,
            "bigode_inferior": min(internos), "bigode_superior": max(internos),
            "outliers": [v for v in valores if v < inferior or v > superior]}


def interpretacao(x):
    """Texto exploratório por diferença média-mediana / desvio populacional.

    O limiar |diferença|/desvio <= 0,1 é uma heurística descritiva, não um
    teste de simetria ou normalidade e não estabelece causalidade.
    """
    valores = _dados(x)
    dp = desvio_padrao(valores, False)
    candidatos = len(outliers_iqr(valores)["outliers"])
    if dp == 0:
        formato = "Todos os valores são iguais; não há dispersão."
    else:
        diferenca = (media(valores) - mediana(valores)) / dp
        if abs(diferenca) <= 0.1:
            formato = "Média e mediana estão próximas em relação à dispersão. Isso não prova simetria ou normalidade."
        elif diferenca > 0:
            formato = "A média supera a mediana, sugerindo assimetria à direita; confirme pelo histograma."
        else:
            formato = "A média está abaixo da mediana, sugerindo assimetria à esquerda; confirme pelo histograma."
    return (f"{formato} A regra de 1,5 × IQR sinaliza {candidatos} de {len(valores)} "
            "observações como possíveis outliers. São sinais exploratórios, não erros comprovados; "
            "os dados não foram removidos. A interpretação de assimetria é uma heurística, não um teste estatístico.")


def normal_pdf(x, mu, sigma):
    """Densidade normal exp(-0,5((x-μ)/σ)²)/(σ√(2π)), σ>0."""
    valor, centro, dp = _numero(x, "x"), _numero(mu, "mu"), _numero(sigma, "sigma")
    if dp <= 0:
        raise ValueError("sigma deve ser maior que zero.")
    z = (valor - centro) / dp
    return math.exp(-0.5 * z * z) / (dp * math.sqrt(2 * math.pi))


def uniforme_pdf(x, a, b):
    """Densidade uniforme: 1/(b-a) em [a,b], zero fora; exige a<b."""
    valor, inicio, fim = _numero(x, "x"), _numero(a, "a"), _numero(b, "b")
    if inicio >= fim:
        raise ValueError("A distribuição uniforme exige a < b.")
    return 1.0 / (fim - inicio) if inicio <= valor <= fim else 0.0


def exponencial_pdf(x, taxa):
    """Densidade exponencial: λexp(-λx) para x>=0; zero fora; λ>0."""
    valor, parametro = _numero(x, "x"), _numero(taxa, "taxa")
    if parametro <= 0:
        raise ValueError("taxa deve ser maior que zero.")
    return parametro * math.exp(-parametro * valor) if valor >= 0 else 0.0


def simular_lgn(n, seed=42):
    """n lançamentos de moeda justa: frequência de caras acumulada Sₖ/k.

    Usa gerador local reproduzível, sem alterar o estado aleatório global.
    Uma realização finita não garante aproximação monotônica a 0,5.
    """
    total = _inteiro_positivo(n, "n")
    rng, caras, frequencias_acumuladas = random.Random(seed), 0, []
    for tentativa in range(1, total + 1):
        caras += int(rng.random() < 0.5)
        frequencias_acumuladas.append(caras / tentativa)
    return frequencias_acumuladas


def simular_tcl(dados, tamanho, repeticoes, seed=42):
    """Médias de amostras independentes com reposição da população empírica.

    Retorna 'repeticoes' médias, cada uma com 'tamanho' sorteios. O tamanho
    pode superar a quantidade de dados, pois a amostragem é com reposição.
    """
    valores = _dados(dados)
    n = _inteiro_positivo(tamanho, "tamanho")
    r = _inteiro_positivo(repeticoes, "repeticoes")
    rng = random.Random(seed)
    return [_media([rng.choice(valores) for _ in range(n)]) for _ in range(r)]
