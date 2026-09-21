"""Execute: python -m streamlit run app.py. Interface separada da matemática."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import minhastats as ms
import graficos as g
from dados import carregar_dados, NUMERICAS, CATEGORICAS, FONTE_URL
from analise import descobertas, textos_descobertas, graficos_descobertas

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Pedal em Dados", page_icon="🚲", layout="wide")
st.markdown("""<style>.block-container{max-width:1200px;padding-top:2rem}
h1{letter-spacing:-1.3px}div[data-testid="stMetric"]{background:#eaf1f3;padding:16px;border-radius:12px}
div[data-testid="stSidebar"]{border-right:1px solid #d8e4e8}</style>""", unsafe_allow_html=True)

@st.cache_data
def carregar():
    return carregar_dados()

def exibir(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)

def selecionar(label="Variável numérica", padrao="cnt", key=None):
    return st.selectbox(label, list(NUMERICAS), index=list(NUMERICAS).index(padrao),
                        format_func=NUMERICAS.get, key=key)

def numero(valor):
    if valor is None:
        return "Indefinido"
    if isinstance(valor, list):
        return ", ".join(f"{x:g}" for x in valor[:12]) + (f" … ({len(valor)} valores)" if len(valor)>12 else "")
    return f"{valor:,.5f}".rstrip("0").rstrip(".")

try:
    df = carregar()
except (ValueError, FileNotFoundError) as erro:
    st.error(f"Não foi possível carregar a base: {erro}")
    st.stop()

MODULOS = ["0 · Conheça os dados", "1 · Núcleo estatístico", "2 · Explore as variáveis",
           "3 · Probabilidade e simulação", "4 · Distribuições teóricas", "5 · Correlação e regressão",
           "6 · Três descobertas"]
st.sidebar.markdown("## 🚲 Pedal em Dados")
st.sidebar.caption("LABORATÓRIO ESTATÍSTICO INTERATIVO")
pagina = st.sidebar.radio("Percurso do laboratório", MODULOS, key="modulo")
st.sidebar.divider()
st.sidebar.markdown("**Projeto individual**  \nMatemática e Estatística para Computação")
st.sidebar.caption("Capital Bikeshare · 2011–2012\n\nMedidas calculadas pelo núcleo minhastats.")
st.sidebar.link_button("Fonte original na UCI", FONTE_URL)
st.title("Pedal em Dados")
st.caption("Investigue como o contexto de uma hora se relaciona com o uso de bicicletas compartilhadas.")
st.header(pagina)

if pagina == MODULOS[0]:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Registros horários", f"{len(df):,}".replace(",", "."))
    c2.metric("Variáveis numéricas", len(NUMERICAS))
    c3.metric("Categorias disponíveis", len(CATEGORICAS))
    c4.metric("Período 2011–2012", "2 anos")
    st.subheader("Uma hora, um retrato da demanda")
    st.write("Cada linha reúne os aluguéis de uma hora e suas condições meteorológicas e de calendário. "
             "A base pública permite comparar dias úteis e folgas, investigar a temperatura e reconhecer picos de uso.")
    st.info("A base original tem 17 colunas. Códigos categóricos foram rotulados; nenhum registro foi imputado ou removido. "
            "As variáveis meteorológicas permanecem na escala normalizada original.")
    st.dataframe(df.head(12), hide_index=True, width="stretch")
    st.subheader("Dicionário das variáveis usadas")
    st.dataframe(pd.DataFrame([{"Coluna": k,"Descrição": v,"Tipo": "Numérica"} for k,v in NUMERICAS.items()] +
                             [{"Coluna": k,"Descrição": v,"Tipo": "Categórica"} for k,v in CATEGORICAS.items()]),
                 hide_index=True, width="stretch")
    st.caption("A contagem do CSV é 17.379; o cabeçalho do catálogo UCI informa 17.389. A análise utiliza o arquivo efetivamente carregado. "
               "Registros são horas, não pessoas; existem horas ausentes no período, que não foram tratadas como demanda zero.")
    st.download_button("Baixar CSV original", (ROOT/"dados/dataset.csv").read_bytes(), "dataset.csv", "text/csv")

elif pagina == MODULOS[1]:
    st.write("Todas as medidas exibidas são calculadas em Python puro. NumPy e SciPy entram nos testes de validação.")
    col = selecionar()
    resumo = ms.resumo(df[col].tolist())
    st.dataframe(pd.DataFrame([{"Medida": k.replace("_"," "), "Resultado": numero(v)} for k,v in resumo.items()]),
                 hide_index=True, width="stretch")
    st.latex(r"\bar{x}=\frac{1}{n}\sum x_i \qquad s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}")
    st.caption("Variância amostral divide por n−1; populacional por n. Percentis usam interpolação linear. "
               "CV = 100 × s / |média|; média próxima de zero torna a medida indefinida. Moda retorna todos os empates.")
    with st.expander("Veja a implementação do núcleo"):
        st.code((ROOT/"minhastats.py").read_text(encoding="utf-8"), language="python")
    validacao = ROOT/"docs/validacao.csv"
    if validacao.exists():
        with st.expander("Comparação numérica com as bibliotecas de referência"):
            st.dataframe(pd.read_csv(validacao), hide_index=True, width="stretch")
            st.caption("Tolerância: |próprio − referência| ≤ 1e−10 + 1e−9 × |referência|. Os casos de borda estão nos testes.")

elif pagina == MODULOS[2]:
    tipo = st.radio("Tipo de variável", ["Numérica", "Categórica"], horizontal=True)
    if tipo == "Numérica":
        col = selecionar()
        valores = df[col].tolist()
        a,b,c = st.columns(3)
        a.metric("Média", numero(ms.media(valores)))
        b.metric("Mediana", numero(ms.mediana(valores)))
        c.metric("Desvio amostral", numero(ms.desvio_padrao(valores)))
        st.info(ms.interpretacao(valores))
        exibir(g.histograma(valores, NUMERICAS[col])[0])
        exibir(g.boxplot(valores, NUMERICAS[col]))
        o = ms.outliers_iqr(valores)
        st.write(f"**{len(o['outliers'])} registros sinalizados** fora de [{o['limite_inferior']:.4f}; {o['limite_superior']:.4f}]. "
                 "São valores atípicos pela regra IQR; isso não determina que sejam erros.")
        tabela = pd.DataFrame(ms.frequencias(valores))
        tabela["relativa (%)"] = tabela.pop("relativa") * 100
        st.subheader("Tabela de frequências")
        st.caption("Classes de Sturges. Intervalos fechados à esquerda e abertos à direita, exceto o último, que inclui o máximo. "
                   "Acumulada é contagem; relativa é percentual. As classes também agrupam as contagens discretas da base.")
        st.dataframe(tabela, hide_index=True, width="stretch")
    else:
        col = st.selectbox("Variável categórica", list(CATEGORICAS), format_func=CATEGORICAS.get)
        valores = df[col].tolist()
        exibir(g.categorias(valores, CATEGORICAS[col]))
        tabela = pd.DataFrame(ms.frequencias_categoricas(valores))
        tabela["relativa (%)"] = tabela.pop("relativa") * 100
        st.dataframe(tabela, hide_index=True, width="stretch")
        st.caption("A ordem das categorias serve à apresentação; frequências acumuladas não indicam uma ordem natural entre elas.")

elif pagina == MODULOS[3]:
    seed = st.number_input("Semente aleatória (reproduz a experiência)", min_value=0, max_value=999999, value=42, step=1)
    st.subheader("A · Lei dos Grandes Números")
    n = st.slider("Número de lançamentos da moeda", 100, 20000, 5000, 100)
    f = ms.simular_lgn(n, seed=int(seed))
    exibir(g.lgn(f))
    st.write(f"Frequência final de caras: **{f[-1]:.4f}**. A referência é 0,5. A convergência não precisa ser monotônica; "
             "uma nova sequência pode oscilar de outro modo.")
    st.subheader("B · Teorema Central do Limite")
    col = selecionar("Variável para as amostras", "casual")
    c1,c2 = st.columns(2)
    tamanho = c1.slider("Tamanho de cada amostra", 2, 150, 30)
    repeticoes = c2.slider("Número de repetições", 100, 3000, 1000, 100)
    valores = df[col].tolist()
    medias = ms.simular_tcl(valores, tamanho, repeticoes, seed=int(seed))
    exibir(g.tcl(valores, medias, tamanho, NUMERICAS[col]))
    st.write(f"Média das médias: **{ms.media(medias):.4f}**; média da base: **{ms.media(valores):.4f}**. "
             f"Desvio das médias: **{ms.desvio_padrao(medias):.4f}**; referência σ/√n: "
             f"**{ms.desvio_padrao(valores, False)/tamanho**.5:.4f}**.")
    st.info("Compare n=2 com n=30 e n=100. Cada amostra é sorteada com reposição da base, "
            "independentemente. Com n crescente, as médias tendem à forma Normal e sua dispersão diminui. "
            "As horas originais têm dependência temporal; esta simulação demonstra o TCL sobre a distribuição empírica, "
            "sem afirmar que as observações cronológicas são independentes.")

elif pagina == MODULOS[4]:
    col = selecionar(padrao="temp")
    valores = df[col].tolist()
    modelos = ["Uniforme"] + (["Exponencial"] if min(valores)>=0 and ms.media(valores)>0 else [])
    modelo = st.selectbox("Compare a Normal com", modelos)
    exibir(g.distribuicoes(valores, NUMERICAS[col], modelo))
    st.write(f"Parâmetros Normal: μ = **{ms.media(valores):.4f}**, σ populacional = **{ms.desvio_padrao(valores, False):.4f}**.")
    if modelo == "Uniforme":
        st.write(f"Uniforme estimada no intervalo **[{min(valores):.4f}; {max(valores):.4f}]**. "
                 "Sua densidade constante contrasta com concentrações ou picos do histograma.")
    else:
        st.write(f"Taxa Exponencial λ = 1/média = **{1/ms.media(valores):.4f}**. "
                 "A densidade é máxima em zero e decresce; picos distantes de zero indicam desacordo com esse modelo.")
    st.info(ms.interpretacao(valores) + " Compare a posição dos picos, a largura e as caudas. "
            "A inspeção visual não é um teste formal de aderência.")
    if col in ["cnt","casual","registered"]:
        st.warning("Aluguéis são contagens discretas. Estas densidades contínuas são aproximações visuais; "
                   "não devem ser interpretadas como probabilidades exatas de uma contagem.")
    else:
        st.caption("As variáveis meteorológicas têm suporte limitado na codificação da base. A Normal atribui massa fora "
                   "desse suporte; mesmo uma curva visualmente próxima não representa perfeitamente esses dados.")

elif pagina == MODULOS[5]:
    c1,c2 = st.columns(2)
    with c1:
        cx = selecionar("Variável X", "temp", "x")
    with c2:
        cy = st.selectbox("Variável Y", [c for c in NUMERICAS if c != cx],
                          index=[c for c in NUMERICAS if c != cx].index("cnt") if cx != "cnt" else 0,
                          format_func=NUMERICAS.get, key="y")
    x,y = df[cx].tolist(), df[cy].tolist()
    b0,b1,r2 = ms.regressao_linear(x,y)
    a,b = st.columns(2)
    a.metric("Correlação de Pearson (r)", f"{ms.correlacao(x,y):.4f}")
    b.metric("R² na própria base", f"{r2:.4f}")
    exibir(g.regressao(x,y,NUMERICAS[cx],NUMERICAS[cy]))
    st.latex(r"\hat{y}="+f"{b0:.4f}"+f"+({b1:.4f})x")
    st.write(f"Cada unidade a mais de X está associada, na reta ajustada, a uma variação média de **{b1:.4f}** unidades de Y. "
             f"O intercepto **{b0:.4f}** é o valor da reta em X=0; fora da faixa observada, essa interpretação é uma extrapolação.")
    valor = st.number_input(f"Valor de X para predição (faixa {min(x):g} a {max(x):g})", value=float(ms.media(x)), format="%.4f")
    st.metric("Predição pela reta", f"{b0+b1*valor:.4f}")
    if not min(x)<=valor<=max(x):
        st.warning("Extrapolação: X está fora da faixa observada. A previsão pode ser inadequada.")
    if cy in ["cnt","casual","registered"] and b0+b1*valor<0:
        st.warning("A reta produziu uma contagem negativa, que não é fisicamente possível. Isso expõe um limite do modelo linear.")
    st.info("Correlação não implica causalidade. Clima, calendário e horário podem influenciar a associação. "
            "R² descreve o ajuste nesta base e não valida previsões futuras.")
    if {cx,cy}.intersection({"cnt"}) and {cx,cy}.intersection({"casual","registered"}):
        st.warning("O total é casual + registered. Correlacionar uma parte com o total cria uma associação estrutural; "
                   "isso não constitui uma descoberta causal.")

elif pagina == MODULOS[6]:
    resultado = descobertas(df)
    textos = textos_descobertas(resultado)
    figs = graficos_descobertas(df, resultado)
    for i, ((titulo,texto),fig) in enumerate(zip(textos,figs),1):
        st.subheader(f"{i}. {titulo}")
        st.write(texto)
        exibir(fig)
    st.caption("As descobertas descrevem o sistema Capital Bikeshare em 2011–2012. Não representam automaticamente "
               "outras cidades ou a demanda atual. Não há avaliação causal ou de desempenho futuro.")
    st.download_button("Baixar evidências em JSON", json.dumps(resultado,ensure_ascii=False,indent=2), "descobertas.json", "application/json")
    relatorio = ROOT/"RELATORIO.md"
    if relatorio.exists():
        st.download_button("Baixar relatório", relatorio.read_bytes(), "RELATORIO.md", "text/markdown")
