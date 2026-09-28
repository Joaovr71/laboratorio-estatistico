"""PDF de envio: preencha entrega.json e execute python scripts/gerar_pdf.py.

Use --modelo para gerar uma versão identificada como incompleta.
"""
import argparse
import json
import re
import sys
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dados import carregar_dados, FONTE_URL
from analise import descobertas, textos_descobertas

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", action="store_true")
    args = parser.parse_args()
    identidade = json.loads((ROOT/"entrega.json").read_text(encoding="utf-8"))
    campos = ["nome", "matricula", "nome_projeto", "repositorio", "video"]
    pendentes = [c for c in campos if not str(identidade.get(c, "")).strip() or "PREENCHER" in str(identidade.get(c, "")).upper()]
    for chave in ["repositorio", "video"]:
        endereco = urlparse(str(identidade.get(chave, "")))
        if endereco.scheme != "https" or not endereco.netloc:
            if chave not in pendentes:
                pendentes.append(chave)
    if pendentes and not args.modelo:
        parser.error("Complete entrega.json antes da versão final: " + ", ".join(pendentes) + ". Para uma prévia, use --modelo.")
    nome = "SISTEMATIZACAO_MEC_PedalEmDados" + ("_MODELO" if args.modelo else "") + ".pdf"
    destino = ROOT/nome
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TituloPedal", fontName="Helvetica-Bold", fontSize=28, leading=32,
                              textColor=colors.HexColor("#132C3D"), spaceAfter=14))
    styles.add(ParagraphStyle("SecaoPedal", fontName="Helvetica-Bold", fontSize=13, leading=17,
                              textColor=colors.HexColor("#147D73"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle("CorpoPedal", fontName="Helvetica", fontSize=10.1, leading=14.2,
                              textColor=colors.HexColor("#253B4A"), spaceAfter=8))
    styles.add(ParagraphStyle("PequenoPedal", fontName="Helvetica", fontSize=8.3, leading=11,
                              textColor=colors.HexColor("#425B6A"), spaceAfter=7))
    def p(texto, estilo="CorpoPedal"):
        return Paragraph(texto, styles[estilo])
    def safe(chave):
        valor = str(identidade.get(chave,"PREENCHER"))
        return escape(valor) if valor and "PREENCHER" not in valor.upper() else "PREENCHER"
    def link(chave):
        valor = str(identidade.get(chave,""))
        if chave in pendentes:
            return "PREENCHER - ainda não informado"
        return f'<link href="{escape(valor,quote=True)}" color="#147D73">{escape(valor)}</link>'
    story = [p("PEDAL EM DADOS", "TituloPedal"), p("Laboratório Estatístico Interativo", "SecaoPedal"),
             p("Matemática e Estatística para Computação | Projeto individual")]
    if args.modelo:
        story += [p("MODELO PARA PREENCHIMENTO - NÃO ENVIAR AO AVA", "SecaoPedal"),
                  p("Este arquivo é uma prévia. Complete a identificação e os links em entrega.json, grave o vídeo e gere a versão final.")]
    story += [Spacer(1,.4*cm),p("1. Identificação", "SecaoPedal"),
              p(f"<b>Nome completo:</b> {safe('nome')}"),p(f"<b>Matrícula:</b> {safe('matricula')}"),
              p(f"<b>Nome do projeto:</b> {safe('nome_projeto')}"),p("<b>Composição:</b> um estudante; desenvolvimento individual."),
              p("2. Links da entrega", "SecaoPedal"),
              p(f'<b>Dados originais:</b><br/><link href="{FONTE_URL}" color="#147D73">{FONTE_URL}</link>'),
              p(f"<b>Repositório público:</b><br/>{link('repositorio')}"),
              p(f"<b>Vídeo de demonstração (3 a 5 minutos):</b><br/>{link('video')}"),
              p("3. Conferência antes do envio", "SecaoPedal"),
              p("Verifique os links do repositório e do vídeo em uma janela anônima. O repositório deve conter código, dados, testes, README, relatório e capturas reais. "
                "O histórico de desenvolvimento deve registrar seu trabalho real; este pacote não comprova esse histórico."),
              p("O resumo executivo da próxima página apresenta o dataset, os sete módulos e as três descobertas reproduzidas pela aplicação."), PageBreak(),
              p("Resumo executivo", "TituloPedal"),
              p("Pedal em Dados | Projeto individual", "PequenoPedal"),
              p("Objetivo e base", "SecaoPedal"),
              p("Explorar a relação entre contexto meteorológico, calendário e demanda por bicicletas compartilhadas. "
                "O arquivo horário Bike Sharing da UCI contém 17.379 registros de 2011 e 2012, do sistema Capital Bikeshare, "
                "com 17 colunas originais. A aplicação oferece sete variáveis numéricas e quatro categóricas rotuladas. "
                "Os dados originais foram preservados, sem imputação, remoção de atípicos ou preenchimento de horas ausentes."),
              p("Método e módulos implementados", "SecaoPedal"),
              p("<b>0.</b> Carregamento e validação da base. <b>1.</b> Núcleo próprio de medidas descritivas, covariância e Pearson, "
                "validado contra NumPy/SciPy. <b>2.</b> Frequências, histogramas, boxplot, IQR e interpretação automática. "
                "<b>3.</b> LGN com moeda e TCL com amostragem da base, controles e semente. <b>4.</b> Normal comparada à Uniforme "
                "ou Exponencial. <b>5.</b> Regressão por mínimos quadrados, R² e predição. <b>6.</b> Descobertas com números e gráficos."),
              p("Três descobertas", "SecaoPedal")]
    resultado = descobertas(carregar_dados())
    for i,(titulo,texto) in enumerate(textos_descobertas(resultado),1):
        story += [p(f"<b>{i}. {escape(titulo)}</b><br/>{escape(texto)}")]
    story += [p("Limites e reprodutibilidade", "SecaoPedal"),
              p("As associações não demonstram causalidade e o R² foi calculado na própria base. Há dependência temporal; "
                "o TCL usa sorteios independentes da distribuição empírica. Os resultados não se generalizam automaticamente "
                "para outras cidades ou períodos. requirements.txt, testes, semente e scripts permitem reproduzir a análise."),
              p("Fonte: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. DOI: 10.24432/C5W894. "
                "Dados sob CC BY 4.0. O total utilizado é a contagem do CSV, que difere do cabeçalho do catálogo.","PequenoPedal")]
    def rodape(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D8E4E8"))
        canvas.line(1.8*cm,1.35*cm,A4[0]-1.8*cm,1.35*cm)
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#425B6A"))
        canvas.drawString(1.8*cm,.9*cm,"Pedal em Dados | " + ("Modelo para preenchimento" if args.modelo else "Sistematização MEC"))
        canvas.drawRightString(A4[0]-1.8*cm,.9*cm,str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(destino),pagesize=A4,rightMargin=1.8*cm,leftMargin=1.8*cm,
                            topMargin=1.7*cm,bottomMargin=1.8*cm,title="Pedal em Dados - Sistematização MEC")
    doc.build(story,onFirstPage=rodape,onLaterPages=rodape)
    print(f"PDF criado: {destino}")
    if pendentes:
        print("Campos pendentes: " + ", ".join(pendentes))

if __name__ == "__main__":
    main()
