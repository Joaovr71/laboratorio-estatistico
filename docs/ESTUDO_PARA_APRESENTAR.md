# Guia de estudo para explicar o projeto

Use este material com a aplicação aberta. Para cada conceito, encontre a função correspondente em `minhastats`, teste um pequeno exemplo à mão e observe um caso real no aplicativo. O objetivo é conseguir explicar o cálculo e a interpretação, inclusive suas limitações.

## O que o projeto investiga

O laboratório usa observações horárias de bicicletas compartilhadas para explorar distribuições, relações entre variáveis e simulações. Cada linha é uma observação horária. `cnt` é o total de aluguéis; `casual` e `registered` são componentes desse total. Uma categoria codificada com números, como a condição do tempo, continua sendo uma categoria.

Os dados ambientais incluem valores normalizados, preservados neste projeto. Antes de interpretar um coeficiente de regressão ou informar uma previsão, confira a escala indicada na interface. Não atribua graus Celsius a um número normalizado. O projeto evita converter temperaturas porque as descrições da fonte divergem; também preserva a estação como código original sem inferir rótulos.

## Fórmulas que você deve conseguir explicar

Nas expressões abaixo, $n$ é a quantidade de observações, $x_i$ é a observação de índice $i$ e $\bar{x}$ é a média.

| Medida | Fórmula / regra | Como explicar |
|---|---|---|
| Média | $\bar{x}=\frac{1}{n}\sum_i x_i$ | Soma dos valores dividida pela quantidade. É sensível a extremos. |
| Mediana | Valor central dos dados ordenados; média dos dois centrais se $n$ for par. | Divide a amostra ordenada ao meio. |
| Moda | Valor ou valores de maior frequência. | O núcleo devolve todos os valores empatados; se todos são distintos, todos aparecem como empate de frequência 1. |
| Amplitude | $\max(x)-\min(x)$ | Distância entre o maior e o menor valor. |
| Variância populacional | $\sigma^2=\frac{1}{n}\sum_i(x_i-\bar{x})^2$ | Média dos desvios quadráticos para o conjunto tratado como população. |
| Variância amostral | $s^2=\frac{1}{n-1}\sum_i(x_i-\bar{x})^2$ | Usa a correção de Bessel quando a amostra estima uma variância populacional. Exige $n\ge2$. |
| Desvio padrão | $s=\sqrt{s^2}$ | Retorna à unidade dos dados; resume dispersão em torno da média. |
| Coeficiente de variação | $CV=100\frac{s}{\lvert\bar{x}\rvert}\%$ | Dispersão relativa com módulo da média. O núcleo rejeita média zero ou muito próxima de zero. Faz sentido sobretudo em escalas de razão. |
| IQR | $Q_3-Q_1$ | Largura dos 50% centrais dos dados. |
| Limites de extremos | $Q_1-1{,}5IQR$ e $Q_3+1{,}5IQR$ | Valores além desses limites são sinalizados para investigação. |

**Quantis com interpolação linear:** ordene os valores, use $h=(n-1)p$, em que $p\in[0,1]$, e interpole entre as posições vizinhas. Para `[0, 10, 20, 30]` e $p=0{,}25$, $h=0{,}75$, portanto $Q_1=7{,}5$. Quartis são os quantis 25%, 50% e 75%. Bibliotecas podem adotar convenções diferentes; por isso a validação precisa usar a mesma convenção.

**Classes de frequência:** a regra de Sturges sugere $k=\lceil1+\log_2(n)\rceil$ classes. Ela é uma escolha inicial, e não uma regra universal sobre a melhor visualização. O código inclui o limite inferior e exclui o superior em cada classe, exceto na última, que inclui os dois limites. A soma das frequências deve ser $n$.

## Covariância, correlação e regressão

A covariância amostral é:

$$
s_{xy}=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{n-1}.
$$

Seu sinal indica como as variáveis tendem a variar juntas, mas sua magnitude depende das unidades. Pearson remove essa dependência:

$$
r=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2\sum_i(y_i-\bar{y})^2}}.
$$

Pearson varia entre −1 e 1 e resume associação **linear**. Um valor perto de zero pode coexistir com uma relação não linear. Quando uma variável é constante, o denominador é zero e a correlação não está definida.

A regressão linear simples ajusta $\hat{y}=b_0+b_1x$, com:

$$
b_1=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2},\qquad b_0=\bar{y}-b_1\bar{x}.
$$

O ajuste minimiza a soma dos resíduos quadráticos, $SSE=\sum_i(y_i-\hat y_i)^2$. O coeficiente de determinação é $R^2=1-SSE/SST$, em que $SST=\sum_i(y_i-\bar y)^2$. O núcleo rejeita Y constante, pois $SST=0$, e X constante, pois não há variação para estimar a inclinação.

Na regressão simples com intercepto, calculada na mesma amostra e sem degeneração, $R^2=r^2$. Isso é uma propriedade desse ajuste, não uma identidade para qualquer modelo. O R² calculado no ajuste não mede, sozinho, desempenho em dados futuros.

Uma previsão usa o valor digitado de X na equação da reta. Extrapolar além da faixa observada aumenta a incerteza. Previsões negativas de contagens podem surgir em uma reta simples e sinalizam uma limitação do modelo. Nenhuma dessas medidas estabelece que alterar X causará uma mudança em Y.

## Entender as duas simulações

**Lei dos Grandes Números (LGN):** em lançamentos independentes de uma moeda com probabilidade fixa $p$, a frequência relativa tende a se aproximar de $p$ conforme a quantidade de lançamentos cresce. O laboratório usa moeda justa, com $p=0{,}5$. Não existe garantia de aproximação monotônica: o erro pode aumentar em trechos da simulação.

**Teorema Central do Limite (TCL):** para amostras independentes de uma distribuição com variância finita, a média amostral padronizada tende a uma distribuição Normal conforme o tamanho amostral cresce. A média das médias se aproxima de $\mu$, e o desvio padrão das médias é $\sigma/\sqrt{n}$ sob as condições usuais.

No laboratório, o sorteio com reposição da variável escolhida cria amostras a partir da distribuição empírica do dataset. Isso ilustra o TCL sobre essa distribuição. Os registros originais são temporais e podem depender uns dos outros; a simulação não demonstra que as horas observadas eram independentes.

O controle de **tamanho da amostra** muda quantos valores entram em cada média. O controle de **repetições** muda quantas médias aparecem no histograma. A **semente** permite repetir os sorteios na mesma configuração; ela não elimina a aleatoriedade do modelo nem torna uma experiência isolada uma prova matemática.

## Comparar distribuições teóricas

| Distribuição | Forma e suporte | Parâmetros e limites de interpretação |
|---|---|---|
| Normal | Contínua, simétrica, definida em toda a reta. | Localização pela média e escala pelo desvio padrão. Pode atribuir probabilidade a valores negativos, inadequados para certas medidas. |
| Uniforme | Contínua, densidade constante entre dois limites. | Limites estimados pelo mínimo e máximo quando essa é a convenção adotada. Estar entre limites não torna os dados uniformes. |
| Exponencial | Contínua, assimétrica à direita, com suporte não negativo. | Para localização fixada em zero, $\lambda=1/\bar{x}$. Exige média positiva e compatibilidade do suporte. Não modela automaticamente toda variável assimétrica. |

O histograma precisa estar em **densidade**, com área total próxima de 1, para a curva teórica ser comparável. A altura de uma densidade não é a probabilidade de um ponto. Uma contagem discreta, como `cnt`, pode ser visualizada com uma aproximação contínua, mas essa aproximação deve ser declarada e criticada. A avaliação visual não substitui um teste formal de aderência.

## Perguntas comuns na apresentação

**Por que usar dados reais?** Para aplicar os conceitos em um contexto observável e interpretar resultados que têm unidade, origem e limitações.

**Por que não usar apenas as funções estatísticas do pandas?** O objetivo didático inclui implementar e compreender o núcleo matemático. Bibliotecas de referência ajudam a validar esse núcleo; a interface deve exibir as medidas da implementação própria.

**Por que dividir por n − 1?** Porque estimar a média a partir da amostra consome um grau de liberdade; sob as hipóteses usuais, essa correção evita o viés para baixo do estimador da variância populacional.

**Um outlier é um erro?** Não. O IQR sinaliza observações que merecem análise. Uma hora com demanda muito alta pode ser real; removê-la sem justificativa distorceria o estudo.

**Correlação alta significa causa?** Não. Pode haver fatores de confusão, tendência temporal, relação estrutural entre medidas ou outros mecanismos. `registered` participa do cálculo de `cnt`, então a associação entre eles exige cuidado especial.

**O que é uma descoberta sustentada por dados?** Uma frase interpretável acompanhada de medida numérica, grupo ou variável analisada e gráfico. “Existe diferença” é insuficiente; explique entre quais grupos, em que medida e de que tamanho.

**Por que os testes usam tolerância?** Cálculos com ponto flutuante podem diferir em pequenas casas decimais devido à representação e à ordem das operações. A tolerância deve ser pequena, explícita e coerente com a precisão necessária.

**Um teste que passou prova que tudo está correto?** Não. Os testes cobrem os cenários escritos. A interpretação, os rótulos, as hipóteses e os casos que não foram contemplados ainda precisam de revisão.

**O que você fez no trabalho?** Descreva com precisão o que estudou, executou, modificou e interpretou. Se usou assistência de IA, declare conforme as regras da instituição e não atribua a si etapas que não realizou.

## Exercício rápido antes da gravação

1. Calcule média, mediana e variância amostral de `[1, 2, 3]`: os resultados são 2, 2 e 1.
2. Explique por que duas colunas constantes não têm correlação de Pearson definida.
3. Troque a variável no Módulo 2 e descreva o que mudou na distribuição.
4. Dobre o tamanho amostral no TCL e observe a dispersão das médias.
5. Faça uma previsão no Módulo 5 e identifique a unidade de X e Y.
6. Leia cada descoberta do relatório, localize seu gráfico e explique uma limitação.

Se alguma explicação parecer decorada, volte ao exemplo pequeno e ao trecho do código. A apresentação fica mais clara quando você consegue relacionar fórmula, implementação, gráfico e conclusão.
