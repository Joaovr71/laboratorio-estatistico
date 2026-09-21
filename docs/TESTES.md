# Registro de validação

A versão entregue do projeto foi verificada no **Windows, com Python 3.12.14** e as versões fixadas em `requirements.txt`.

## Suíte automatizada

**Resultado: 351 testes passaram, em 34,16 segundos.** Esse tempo registra a execução de verificação e não é uma garantia de desempenho em outros computadores.

| Arquivo | Testes aprovados | Principais verificações |
|---|---:|---|
| `test_minhastats.py` | 320 | Medidas próprias contra referências, quantis, empates de moda, variâncias amostral e populacional, covariância, Pearson, regressão, frequências, IQR, densidades, simulações reproduzíveis e tratamento de entradas inválidas. |
| `test_dados.py` | 18 | Dimensões da base real, integridade do arquivo, esquema, domínios, datas, contagens, duplicações e rejeição explícita de dados corrompidos. |
| `test_app.py` | 13 | Abertura dos sete módulos, troca de variáveis e categorias, resposta dos controles de simulação, previsões, alertas de interpretação e uso de cálculos próprios nos gráficos. |
| **Total** | **351** | **Suíte completa aprovada.** |

Os testes do núcleo incluem amostras vazias, valores não finitos, médias próximas de zero, pares incompatíveis, variáveis constantes e parâmetros inválidos. A suíte também verifica que o núcleo usa apenas os módulos permitidos da biblioteca padrão.

## Validação numérica documentada

O script `scripts/gerar_resultados.py` produziu **209 comparações aprovadas** em [validacao.csv](validacao.csv). A maior diferença absoluta registrada foi **7,275957614183426 × 10⁻¹²**.

A condição utilizada para cada componente é:

```text
|valor próprio − referência| ≤ atol + rtol × |referência|
rtol = 1e-9
atol = 1e-10
```

As referências são calculadas com NumPy e SciPy, conforme a função identificada em cada linha do CSV. Quando o resultado é um vetor, seus componentes são comparados individualmente. O arquivo registra o cenário, os valores ou sua representação resumida, a quantidade de componentes, a maior diferença, as tolerâncias e a aprovação.

As 209 comparações do relatório são uma evidência adicional gerada sobre os cenários documentados. Elas são distintas dos 351 casos executados pelo pytest; os dois números não devem ser somados como se representassem uma única suíte.

## Reproduzir no seu computador

Abra o PowerShell na pasta principal do projeto, depois de instalar as dependências conforme o README:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/gerar_resultados.py
```

Para executar apenas uma parte, informe o arquivo, por exemplo:

```powershell
.\.venv\Scripts\python.exe -m pytest -q test_minhastats.py
```

Na preparação do pacote, os arquivos temporários dos testes foram direcionados para uma pasta local por uma restrição de escrita do ambiente. Isso foi uma configuração de execução, sem mudança funcional no projeto. Em um ambiente com a mesma restrição, a opção `--basetemp` do pytest permite escolher uma pasta temporária autorizada.

## Alcance da verificação

Os testes verificam os cenários escritos e a integração automatizada da aplicação. As capturas `modulo_0.png` a `modulo_6.png` documentam a interface em execução; consulte também [o README](../README.md) e [o relatório](../RELATORIO.md).

A aprovação não comprova causalidade, aderência formal a uma distribuição, qualidade de previsão em dados futuros ou adequação de toda interpretação possível. Revise as unidades, hipóteses e conclusões; rode novamente os testes depois de modificar código ou dados. Identificação, publicação no GitHub, gravação do vídeo e acesso aos links continuam sendo etapas pessoais da entrega.
