"""
Gerador de convites de casamento em PDF personalizados.
Lê a lista de convidados do CSV e gera um PDF para cada um.
"""

import csv
import os
from reportlab.lib.pagesizes import A5
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from config import (
    NOIVA, NOIVO, DATA_CASAMENTO, HORARIO_CERIMONIA, HORARIO_FESTA,
    LOCAL_CERIMONIA, ENDERECO_CERIMONIA, LOCAL_FESTA, ENDERECO_FESTA,
    MENSAGEM_ESPECIAL, DRESS_CODE, LINK_PRESENTES, LINK_CONFIRMACAO,
    COR_PRINCIPAL, COR_SECUNDARIA, COR_TEXTO, COR_FUNDO,
)

PASTA_PDF = os.path.join(os.path.dirname(__file__), "convites_pdf")
ARQUIVO_CSV = os.path.join(os.path.dirname(__file__), "convidados.csv")


def cor(rgb):
    return Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def desenhar_borda(c, largura, altura):
    """Desenha uma borda decorativa dourada."""
    margem = 0.8 * cm
    c.setStrokeColor(cor(COR_SECUNDARIA))
    c.setLineWidth(2)
    c.rect(margem, margem, largura - 2 * margem, altura - 2 * margem)
    c.setLineWidth(0.5)
    c.rect(margem + 4, margem + 4, largura - 2 * margem - 8, altura - 2 * margem - 8)


def gerar_convite(nome_convidado, telefone):
    """Gera um PDF de convite personalizado para um convidado."""
    os.makedirs(PASTA_PDF, exist_ok=True)

    nome_arquivo = f"convite_{telefone}.pdf"
    caminho = os.path.join(PASTA_PDF, nome_arquivo)

    largura, altura = A5
    c = canvas.Canvas(caminho, pagesize=A5)

    # Fundo creme
    c.setFillColor(cor(COR_FUNDO))
    c.rect(0, 0, largura, altura, fill=1, stroke=0)

    # Borda decorativa
    desenhar_borda(c, largura, altura)

    y = altura - 2.5 * cm

    # Ornamento superior
    c.setFillColor(cor(COR_SECUNDARIA))
    c.setFont("Helvetica", 18)
    c.drawCentredString(largura / 2, y, "~ \u2661 ~")
    y -= 1.2 * cm

    # Título
    c.setFillColor(cor(COR_PRINCIPAL))
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(largura / 2, y, "CONVITE DE CASAMENTO")
    y -= 1.5 * cm

    # Nomes dos noivos
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(largura / 2, y, f"{NOIVA}")
    y -= 0.8 * cm
    c.setFont("Helvetica", 14)
    c.setFillColor(cor(COR_SECUNDARIA))
    c.drawCentredString(largura / 2, y, "&")
    y -= 0.8 * cm
    c.setFillColor(cor(COR_PRINCIPAL))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(largura / 2, y, f"{NOIVO}")
    y -= 1.3 * cm

    # Linha decorativa
    c.setStrokeColor(cor(COR_SECUNDARIA))
    c.setLineWidth(0.5)
    c.line(3 * cm, y, largura - 3 * cm, y)
    y -= 0.8 * cm

    # Nome do convidado
    c.setFillColor(cor(COR_TEXTO))
    c.setFont("Helvetica", 9)
    c.drawCentredString(largura / 2, y, "Prezado(a)")
    y -= 0.5 * cm
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(cor(COR_PRINCIPAL))
    c.drawCentredString(largura / 2, y, nome_convidado)
    y -= 1 * cm

    # Mensagem especial
    c.setFillColor(cor(COR_TEXTO))
    c.setFont("Helvetica-Oblique", 8)
    # Quebrar texto longo
    palavras = MENSAGEM_ESPECIAL.split()
    linhas = []
    linha_atual = ""
    for palavra in palavras:
        teste = f"{linha_atual} {palavra}".strip()
        if c.stringWidth(teste, "Helvetica-Oblique", 8) < largura - 4 * cm:
            linha_atual = teste
        else:
            linhas.append(linha_atual)
            linha_atual = palavra
    if linha_atual:
        linhas.append(linha_atual)

    for linha in linhas:
        c.drawCentredString(largura / 2, y, linha)
        y -= 0.4 * cm

    y -= 0.4 * cm

    # Data
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(cor(COR_PRINCIPAL))
    c.drawCentredString(largura / 2, y, DATA_CASAMENTO)
    y -= 0.8 * cm

    # Cerimônia
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(cor(COR_TEXTO))
    c.drawCentredString(largura / 2, y, f"Cerim\u00f4nia: {HORARIO_CERIMONIA}")
    y -= 0.4 * cm
    c.setFont("Helvetica", 8)
    c.drawCentredString(largura / 2, y, LOCAL_CERIMONIA)
    y -= 0.35 * cm
    c.setFont("Helvetica", 7)
    c.drawCentredString(largura / 2, y, ENDERECO_CERIMONIA)
    y -= 0.7 * cm

    # Festa
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(largura / 2, y, f"Festa: {HORARIO_FESTA}")
    y -= 0.4 * cm
    c.setFont("Helvetica", 8)
    c.drawCentredString(largura / 2, y, LOCAL_FESTA)
    y -= 0.35 * cm
    c.setFont("Helvetica", 7)
    c.drawCentredString(largura / 2, y, ENDERECO_FESTA)
    y -= 0.7 * cm

    # Dress code
    if DRESS_CODE:
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(cor(COR_SECUNDARIA))
        c.drawCentredString(largura / 2, y, f"Traje: {DRESS_CODE}")
        y -= 0.6 * cm

    # Links
    c.setFont("Helvetica", 7)
    c.setFillColor(cor(COR_TEXTO))
    if LINK_PRESENTES:
        c.drawCentredString(largura / 2, y, f"Lista de presentes: {LINK_PRESENTES}")
        y -= 0.4 * cm
    if LINK_CONFIRMACAO:
        c.drawCentredString(largura / 2, y, f"Confirme presen\u00e7a: {LINK_CONFIRMACAO}")
        y -= 0.4 * cm

    # Ornamento inferior
    y = 1.5 * cm
    c.setFillColor(cor(COR_SECUNDARIA))
    c.setFont("Helvetica", 14)
    c.drawCentredString(largura / 2, y, "~ \u2661 ~")

    c.save()
    return caminho


def gerar_todos():
    """Gera convites para todos os convidados do CSV."""
    with open(ARQUIVO_CSV, "r", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        total = 0
        for linha in leitor:
            telefone = linha["telefone"].strip()
            nome = linha["nome"].strip()
            if not telefone or not nome:
                continue
            caminho = gerar_convite(nome, telefone)
            print(f"  [OK] {nome} -> {caminho}")
            total += 1

    print(f"\n{total} convite(s) gerado(s) na pasta: {PASTA_PDF}")


if __name__ == "__main__":
    print("Gerando convites de casamento em PDF...\n")
    gerar_todos()
