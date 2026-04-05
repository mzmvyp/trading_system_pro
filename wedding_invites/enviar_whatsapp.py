"""
Envio de convites de casamento via WhatsApp Web.
Usa pywhatkit para automação do navegador (sem API oficial).

IMPORTANTE:
- Você precisa estar logado no WhatsApp Web no navegador padrão
- Na primeira execução, escaneie o QR Code
- Mantenha o navegador aberto durante todo o envio
- Não mexa no computador enquanto os envios estão sendo feitos
"""

import csv
import os
import time
import pywhatkit as kit

from config import (
    NOIVA, NOIVO, DATA_CASAMENTO, HORARIO_CERIMONIA, HORARIO_FESTA,
    LOCAL_CERIMONIA, ENDERECO_CERIMONIA, LOCAL_FESTA, ENDERECO_FESTA,
    MENSAGEM_ESPECIAL, DRESS_CODE, LINK_PRESENTES, LINK_CONFIRMACAO,
    INTERVALO_ENVIO_SEGUNDOS,
)

ARQUIVO_CSV = os.path.join(os.path.dirname(__file__), "convidados.csv")
PASTA_PDF = os.path.join(os.path.dirname(__file__), "convites_pdf")
LOG_ENVIO = os.path.join(os.path.dirname(__file__), "log_envios.csv")


def montar_mensagem(nome_convidado):
    """Monta a mensagem de texto do convite para WhatsApp."""
    msg = (
        f"💒 *CONVITE DE CASAMENTO* 💒\n"
        f"\n"
        f"Querido(a) *{nome_convidado}*,\n"
        f"\n"
        f"Com imensa alegria, convidamos você para celebrar o casamento de\n"
        f"\n"
        f"✨ *{NOIVA} & {NOIVO}* ✨\n"
        f"\n"
        f"{MENSAGEM_ESPECIAL}\n"
        f"\n"
        f"📅 *Data:* {DATA_CASAMENTO}\n"
        f"\n"
        f"⛪ *Cerimônia:* {HORARIO_CERIMONIA}\n"
        f"📍 {LOCAL_CERIMONIA}\n"
        f"    {ENDERECO_CERIMONIA}\n"
        f"\n"
        f"🎉 *Festa:* {HORARIO_FESTA}\n"
        f"📍 {LOCAL_FESTA}\n"
        f"    {ENDERECO_FESTA}\n"
    )

    if DRESS_CODE:
        msg += f"\n👔 *Traje:* {DRESS_CODE}\n"

    if LINK_PRESENTES:
        msg += f"\n🎁 *Lista de presentes:* {LINK_PRESENTES}\n"

    if LINK_CONFIRMACAO:
        msg += f"\n✅ *Confirme sua presença:* {LINK_CONFIRMACAO}\n"

    msg += (
        f"\n"
        f"Com carinho,\n"
        f"*{NOIVA} & {NOIVO}* 💕\n"
        f"\n"
        f"_O convite em PDF segue em seguida!_"
    )

    return msg


def formatar_telefone(telefone):
    """Garante que o telefone está no formato +55..."""
    telefone = telefone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if not telefone.startswith("+"):
        if telefone.startswith("55"):
            telefone = "+" + telefone
        else:
            telefone = "+55" + telefone
    return telefone


def registrar_log(telefone, nome, status, erro=""):
    """Registra o resultado do envio no log."""
    existe = os.path.exists(LOG_ENVIO)
    with open(LOG_ENVIO, "a", encoding="utf-8") as f:
        if not existe:
            f.write("telefone,nome,status,erro,timestamp\n")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{telefone},{nome},{status},{erro},{timestamp}\n")


def enviar_convites(modo_teste=False):
    """
    Envia convites para todos os convidados do CSV.

    Args:
        modo_teste: Se True, apenas mostra as mensagens sem enviar.
    """
    with open(ARQUIVO_CSV, "r", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        convidados = [
            (linha["telefone"].strip(), linha["nome"].strip())
            for linha in leitor
            if linha["telefone"].strip() and linha["nome"].strip()
        ]

    total = len(convidados)
    print(f"\n{'='*50}")
    print(f"  ENVIO DE CONVITES DE CASAMENTO - WhatsApp")
    print(f"  {NOIVA} & {NOIVO}")
    print(f"  Total de convidados: {total}")
    print(f"  Intervalo entre envios: {INTERVALO_ENVIO_SEGUNDOS}s")
    print(f"{'='*50}\n")

    if modo_teste:
        print("⚠️  MODO TESTE - Nenhuma mensagem será enviada\n")

    enviados = 0
    erros = 0

    for i, (telefone, nome) in enumerate(convidados, 1):
        tel_formatado = formatar_telefone(telefone)
        mensagem = montar_mensagem(nome)
        pdf_path = os.path.join(PASTA_PDF, f"convite_{telefone}.pdf")

        print(f"[{i}/{total}] Enviando para {nome} ({tel_formatado})...")

        if modo_teste:
            print(f"  Mensagem:\n{mensagem[:100]}...")
            print(f"  PDF: {pdf_path}")
            print(f"  [TESTE] Pular envio\n")
            registrar_log(tel_formatado, nome, "TESTE")
            continue

        try:
            # Envia a mensagem de texto
            kit.sendwhatmsg_instantly(
                phone_no=tel_formatado,
                message=mensagem,
                wait_time=15,
                tab_close=True,
            )
            print(f"  [OK] Mensagem de texto enviada!")

            # Aguarda um pouco antes de enviar o PDF
            time.sleep(5)

            # Envia o PDF se existir
            if os.path.exists(pdf_path):
                kit.sendwhats_image(
                    receiver=tel_formatado,
                    img_path=pdf_path,
                    caption="Convite de Casamento 💒",
                    wait_time=15,
                    tab_close=True,
                )
                print(f"  [OK] PDF do convite enviado!")
            else:
                print(f"  [!] PDF não encontrado: {pdf_path}")

            registrar_log(tel_formatado, nome, "ENVIADO")
            enviados += 1

        except Exception as e:
            print(f"  [ERRO] Falha ao enviar: {e}")
            registrar_log(tel_formatado, nome, "ERRO", str(e))
            erros += 1

        # Intervalo entre envios
        if i < total:
            print(f"  Aguardando {INTERVALO_ENVIO_SEGUNDOS}s antes do próximo envio...")
            time.sleep(INTERVALO_ENVIO_SEGUNDOS)

    print(f"\n{'='*50}")
    print(f"  RESULTADO FINAL")
    print(f"  Enviados: {enviados}/{total}")
    print(f"  Erros: {erros}")
    print(f"  Log salvo em: {LOG_ENVIO}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys

    if "--teste" in sys.argv:
        enviar_convites(modo_teste=True)
    else:
        print("⚠️  ATENÇÃO: Este script vai enviar mensagens reais no WhatsApp!")
        print("   Certifique-se de que:")
        print("   1. Você está logado no WhatsApp Web")
        print("   2. O CSV de convidados está correto")
        print("   3. Os PDFs foram gerados (rode gerar_pdf.py primeiro)")
        print()
        resposta = input("Deseja continuar? (sim/nao): ").strip().lower()
        if resposta in ("sim", "s", "yes", "y"):
            enviar_convites(modo_teste=False)
        else:
            print("Envio cancelado.")
