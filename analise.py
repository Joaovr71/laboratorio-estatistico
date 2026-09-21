"""Descobertas reproduzíveis; todas as medidas passam pelo núcleo próprio."""
import minhastats as ms
import graficos as g

def descobertas(df):
    grupos = []
    for codigo, nome in [(0, "Folga / feriado"), (1, "Dia útil")]:
        valores = df.loc[df["workingday"] == codigo, "casual"].tolist()
        grupos.append({"grupo": nome, "n": len(valores), "mediana": ms.mediana(valores), "media": ms.media(valores)})
    x, y = df["temp"].tolist(), df["cnt"].tolist()
    b0, b1, r2 = ms.regressao_linear(x, y)
    out = ms.outliers_iqr(y)
    mascara = (df["cnt"] < out["limite_inferior"]) | (df["cnt"] > out["limite_superior"])
    atipicos = df.loc[mascara]
    pico = atipicos.loc[(atipicos["workingday"] == 1) & atipicos["hr"].isin([17, 18])]
    qtd = len(atipicos)
    return {"grupos": grupos, "temperatura": {"r": ms.correlacao(x, y), "b0": b0, "b1": b1, "r2": r2},
            "outliers": {"n": qtd, "percentual": qtd/len(df)*100, "limite_superior": out["limite_superior"],
                         "pico_dia_util": len(pico), "percentual_pico": len(pico)/qtd*100 if qtd else 0}}

def textos_descobertas(resultado):
    a, b = resultado["grupos"]
    t, o = resultado["temperatura"], resultado["outliers"]
    return [
        ("Usuários casuais mudam com o tipo de dia",
         f"A mediana de aluguéis casuais por hora é {a['mediana']:.0f} nas folgas/feriados (n={a['n']}) "
         f"e {b['mediana']:.0f} nos dias úteis (n={b['n']}). Horário, clima e época do ano também variam entre grupos; "
         "a comparação não isola o efeito de folgar."),
        ("Temperatura ajuda a descrever a demanda, mas deixa muita variação",
         f"Temperatura normalizada e total de aluguéis têm r={t['r']:.4f}. A regressão simples apresenta "
         f"R²={t['r2']:.4f}, ou {t['r2']*100:.1f}% da variação observada. "
         "O resultado é descritivo, ajustado na própria base; não mede acurácia futura nem efeito causal."),
        ("Valores atípicos incluem horários plausíveis de alta demanda",
         f"A regra de 1,5 × IQR sinaliza {o['n']} horas ({o['percentual']:.2f}%) e tem limite superior "
         f"{o['limite_superior']:.1f} aluguéis. Dessas horas, {o['pico_dia_util']} ({o['percentual_pico']:.1f}%) "
         "ocorrem às 17h ou 18h em dias úteis. Ser atípico pela regra não demonstra erro; todos os registros foram mantidos.")]

def graficos_descobertas(df, resultado):
    fig, ax = g.base("Usuários casuais por tipo de dia", "", "Mediana de aluguéis / hora")
    ax.bar([r["grupo"] for r in resultado["grupos"]], [r["mediana"] for r in resultado["grupos"]], color=[g.LARANJA, g.VERDE])
    fig2 = g.regressao(df["temp"].tolist(), df["cnt"].tolist(), "Temperatura normalizada", "Total de aluguéis / hora")
    lim = resultado["outliers"]["limite_superior"]
    horas = df.loc[df["cnt"] > lim, "hr"].tolist()
    fig3, ax3 = g.base("Quando aparecem as horas acima do limite IQR?", "Hora do dia", "Horas sinalizadas")
    tabela = ms.frequencias_categoricas(horas)
    mapa = {r["categoria"]: r["frequencia"] for r in tabela}
    ax3.bar(list(range(24)), [mapa.get(h, 0) for h in range(24)], color=g.VERDE)
    ax3.set_xticks(list(range(0,24,2)))
    return [fig, fig2, fig3]
