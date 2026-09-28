# Pedal em Dados — Laboratório Estatístico

Aplicação interativa em Python para explorar dados reais de aluguel de bicicletas, calcular estatísticas com uma biblioteca própria, simular fenômenos probabilísticos e interpretar regressões. Projeto acadêmico **individual**, baseado no conjunto Bike Sharing da UCI.

## Executar no Windows

O projeto foi desenvolvido e validado no Windows com **Python 3.12.14** e com as versões de dependências definidas em `requirements.txt`.

### 1. Abra o PowerShell na pasta do projeto

Extraia o projeto e abra a pasta no VS Code ou no Explorador de Arquivos. O terminal precisa estar aberto **na mesma pasta em que estão `app.py` e `requirements.txt`**.

Para conferir, execute:

```powershell
dir
```

Na listagem devem aparecer, entre outros, arquivos como `app.py`, `requirements.txt`, `minhastats.py`, `graficos.py`, `dados.py` e `analise.py`.

> **Importante:** não execute os comandos a partir de `C:\Windows\System32`, pois o PowerShell não encontrará os arquivos do projeto.

### 2. Verifique se o Python 3.12 está instalado

Execute:

```powershell
py -0
```

Se aparecer uma versão `3.12`, continue para a próxima etapa. O projeto foi validado com Python 3.12.14; usar essa versão ajuda a manter o ambiente reproduzível.

Se o Python 3.12 não estiver instalado, instale essa versão e abra o terminal novamente antes de continuar.

### 3. Crie o ambiente virtual

Dentro da pasta do projeto, execute:

```powershell
py -3.12 -m venv .venv
```

### 4. Instale as dependências

Execute os comandos abaixo, um por vez:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A ativação do ambiente virtual é opcional, pois os comandos acima chamam diretamente o Python da pasta `.venv`.

### 5. Inicie a aplicação

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Quando o Streamlit iniciar, o terminal exibirá um endereço semelhante a:

```text
Local URL: http://localhost:8501
```

Abra `http://localhost:8501` no navegador. **Não digite `Local URL:` no PowerShell**; essa linha é apenas uma informação exibida pelo Streamlit.

Para encerrar a aplicação, volte ao terminal e pressione `Ctrl+C`.

### Executar os testes

Com o ambiente virtual já criado e as dependências instaladas, execute:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Se o comando `py` não for reconhecido, instale o Python com o Python Launcher para Windows e abra um novo terminal. Para manter a compatibilidade com o ambiente em que o projeto foi validado, utilize Python 3.12.

## Identificação

| Campo | Informação |
|---|---|
| Estudante | Joao Victor Ramos Mascarenhas |
| Matrícula | 72650059 |


## Dados e pergunta do projeto

**Pergunta:** como a demanda por bicicletas varia e como se relaciona com condições ambientais e de calendário?

O arquivo `dados/dataset.csv` contém a versão horária `hour.csv`: **17.379 registros e 17 colunas**. A contagem corresponde ao arquivo utilizado; o cabeçalho da página UCI apresenta uma contagem diferente. Os registros se referem a 2011 e 2012 no sistema Capital Bikeshare. O CSV está incluído para permitir executar a aplicação sem baixar os dados novamente.

Fonte: [Bike Sharing — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Referência: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI. [DOI: 10.24432/C5W894](https://doi.org/10.24432/C5W894). Dados disponibilizados sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

As sete variáveis numéricas analisadas são `temp`, `atemp`, `hum`, `windspeed`, `casual`, `registered` e `cnt`. A aplicação apresenta as categorias `clima`, `tipo_dia`, `ano` e `feriado`, derivadas dos códigos originais `weathersit`, `workingday`, `yr` e `holiday`. Códigos de calendário e identificadores não são tratados como medidas contínuas apenas porque aparecem como números no CSV. As 17 colunas são as do CSV original; os quatro rótulos são acrescentados apenas durante o carregamento.

`temp`, `atemp`, `hum` e `windspeed` estão normalizadas na fonte. O projeto preserva essas escalas: divergências entre a página UCI e o arquivo de descrição original impedem uma conversão de temperatura inequívoca. Pelo mesmo motivo, `season` é preservada como código original, sem rótulos de estação inferidos. `casual` e `registered` são contagens de aluguéis por tipo de usuário, e `cnt` é o total de aluguéis; esses valores não representam pessoas únicas. Confira a unidade e o rótulo mostrados pela aplicação antes de interpretar um valor.

## Os sete módulos

| Módulo | O que explorar |
|---|---|
| 0 — Dados reais | Fonte, dimensões, qualidade, tipos de variáveis e amostra dos registros. |
| 1 — Núcleo estatístico próprio | Média, mediana, moda, amplitude, variâncias, desvios, quantis, coeficiente de variação, covariância e Pearson implementados em `minhastats`. |
| 2 — Estatística descritiva | Escolha de variável, tabela de frequências, classes para variáveis numéricas, histogramas, boxplots, categorias e sinalização de valores extremos por IQR. |
| 3 — Probabilidade e simulação | Lei dos Grandes Números com moeda e Teorema Central do Limite com amostras do dataset; controles de repetições, tamanho amostral e semente. |
| 4 — Distribuições teóricas | Histograma de densidade comparado com Normal, Uniforme ou Exponencial, com parâmetros estimados dos dados e comentários sobre adequação. |
| 5 — Correlação e regressão | Escolha de X e Y, dispersão, Pearson, reta por mínimos quadrados, equação, R² e previsão interativa. |
| 6 — Descobertas | Três conclusões sustentadas por resultados e gráficos, com limites de interpretação. |

O núcleo matemático fica separado da interface. NumPy, pandas e SciPy apoiam a manipulação de dados e as referências de validação; as medidas apresentadas como cálculo próprio vêm de `minhastats`.

## Capturas da interface

**Módulo 0 — apresentação dos dados e da origem da base.**

![Interface do Módulo 0, com apresentação dos dados](docs/modulo_0.png)

**Módulo 5 — exploração de correlação, regressão e previsão.**

![Interface do Módulo 5, com correlação e regressão](docs/modulo_5.png)

As capturas dos sete módulos estão na pasta `docs`.

## Arquivos principais

```text
app.py                       interface interativa
minhastats.py                funções estatísticas próprias
dados/dataset.csv            dados reais incluídos
requirements.txt             dependências
test_*.py                    testes do núcleo, dos dados e da aplicação
RELATORIO.md                  metodologia, validação e descobertas
entrega.json                 identificação e links para a entrega
scripts/gerar_resultados.py   atualização dos resultados documentados
scripts/gerar_pdf.py          geração do documento de entrega
docs/                        evidências e materiais de apoio
```

## Validação executada

A suíte completa passou em **351 testes**, em 34,16 segundos na execução de verificação: 320 do núcleo estatístico, 18 dos dados e 13 da aplicação. O ambiente foi Windows com Python 3.12.14 e as dependências de `requirements.txt`. O tempo é o registro dessa execução e pode variar em outros computadores.

O script de resultados também aprovou **209 comparações numéricas** com NumPy/SciPy. A maior diferença absoluta registrada foi aproximadamente **7,275958 × 10⁻¹²**, dentro das tolerâncias `rtol=1e-9` e `atol=1e-10`.

Consulte [o resumo dos testes](docs/TESTES.md), [a tabela de validação](docs/validacao.csv) e [o relatório](RELATORIO.md). Os arquivos `test_minhastats.py`, `test_dados.py` e `test_app.py` detalham os cenários, incluindo casos como amostras pequenas, variáveis constantes, dados corrompidos e interação com os sete módulos.


```

