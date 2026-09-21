# Fonte e integridade dos dados

Este projeto individual utiliza **Bike Sharing**, de **Hadi Fanaee-T (2013)**,
disponibilizado pelo UCI Machine Learning Repository. São contagens horárias de
aluguéis do sistema Capital Bikeshare, com informações meteorológicas e de
calendário, referentes a 2011 e 2012.

- Página original: <https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset>
- DOI: <https://doi.org/10.24432/C5W894>
- ZIP original: <https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip>
- Licença informada na página UCI: [Creative Commons Attribution 4.0 International — CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Arquivo utilizado: `hour.csv`, distribuído aqui como `dataset.csv`, com seus bytes originais.
- Documentação original: `Readme_original.txt`.
- Proveniência e hashes SHA-256: `metadados.json`, gerado ao preparar o arquivo.

**Referência:** Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5W894.

O arquivo horário possui **17.379 registros e 17 colunas originais**, incluindo
identificador, data e variável-alvo. O catálogo UCI apresenta 17.389 instâncias;
o laboratório usa a contagem efetivamente encontrada no CSV. Não há preenchimento
de horas ausentes, remoção de observações nem geração de dados sintéticos para
compor a base real. Outliers são sinalizados pela aplicação, sem exclusão.

## Variáveis utilizadas

| Coluna original | Rótulo usado | Natureza |
|---|---|---|
| `temp` | Temperatura (normalizada) | Numérica |
| `atemp` | Sensação térmica (normalizada) | Numérica |
| `hum` | Umidade (normalizada) | Numérica |
| `windspeed` | Velocidade do vento (normalizada) | Numérica |
| `casual` | Aluguéis de usuários casuais | Contagem numérica |
| `registered` | Aluguéis de usuários cadastrados | Contagem numérica |
| `cnt` | Total de aluguéis por hora | Contagem numérica; `casual + registered` |
| `weathersit` | Condição do tempo (`clima`) | Categórica; quatro códigos |
| `workingday` | Tipo de dia (`tipo_dia`) | Categórica; dia útil ou fim de semana/feriado |
| `yr` | Ano (`ano`) | Categórica; 2011 ou 2012 |
| `holiday` | Feriado (`feriado`) | Categórica; sim ou não |

Os rótulos categóricos em português são derivados no carregamento; as colunas
originais continuam disponíveis. Os códigos de clima correspondem, em forma
resumida, a 1: limpo/poucas nuvens; 2: névoa/muitas nuvens; 3: chuva/neve leve;
4: chuva/neve forte. A descrição completa está no README original.

O identificador `instant`, a data `dteday` e os códigos de calendário/clima não
entram no seletor de variáveis contínuas: a média de um identificador ou de
categorias codificadas numericamente não representa uma medida substantiva.

**Unidades e divergências na documentação:** a página UCI e o README do ZIP
descrevem de forma diferente a normalização da temperatura/sensação térmica e
a associação entre números e nomes das estações. Por isso, `temp`, `atemp`,
`hum` e `windspeed` são mantidas na escala normalizada original, sem conversão
para °C, porcentagem ou velocidade física. `season` é preservada no CSV e
validada como código de 1 a 4, mas não é traduzida nem usada nos seletores.
As quatro categorias escolhidas acima evitam essa ambiguidade.

## Reprodução e verificações

O CSV acompanha o projeto; o uso normal da aplicação dispensa internet.
Para obter uma nova cópia diretamente da UCI, execute na pasta do projeto:

```text
python scripts/baixar_dados.py
```

Para preparar um ZIP oficial já disponível:

```text
python scripts/baixar_dados.py --arquivo-zip caminho/para/bike-sharing.zip
```

O script acessa a rede apenas quando executado explicitamente sem um ZIP local,
lê exclusivamente `hour.csv` e `Readme.txt`, valida o cabeçalho e a quantidade
de registros e grava o manifesto de origem, data e hashes. Não executa extração
arbitrária de caminhos contidos no ZIP. Ao usar um ZIP local, o manifesto
registra essa condição; o hash documenta os bytes, não atesta sua procedência.

No carregamento, `dados.py` rejeita esquema incorreto, quantidade inesperada
de linhas, nulos, valores não numéricos/não finitos, códigos fora do domínio,
normalizações fora de [0, 1], contagens negativas/fracionárias, divergência
em `cnt = casual + registered`, IDs inválidos/duplicados e datas/horas
duplicadas ou inconsistentes com os códigos de calendário. Erros não são
silenciosamente corrigidos.

As observações horárias têm dependência temporal. Correlações e regressões
exploratórias não estabelecem causalidade. Como `cnt` é definido pela soma de
`casual` e `registered`, associações entre essas variáveis são parcialmente
determinísticas e devem ser interpretadas com essa ressalva.
