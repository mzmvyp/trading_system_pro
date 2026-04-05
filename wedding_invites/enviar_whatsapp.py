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

from config import PDF_CONVITE, INTERVALO_ENVIO_SEGUNDOS

ARQUIVO_CSV = os.path.join(os.path.dirname(__file__), "convidados.csv")
LOG_ENVIO = os.path.join(os.path.dirname(__file__), "log_envios.csv")


def montar_mensagem(nome_convidado):
    """Monta a mensagem personalizada do convite."""
    return (
        f"{nome_convidado}, é com muita alegria que convidamos vocês para participar do nosso casamento! ✨\n"
        f"Estamos vivendo um momento único e nossa felicidade só será completa com a presença de vocês. "
        f"Abaixo, deixamos as informações mais importantes para o nosso grande dia:\n"
        f"\n"
        f"📅 DATA: 25 de julho de 2026 (Sábado)\n"
        f"🕒 HORÁRIO: 17h (A festa vai até às 2h, então preparem-se!)\n"
        f"📍 LOCAL: Casa do Ator (Grupo Bisutti)\n"
        f"🏠 ENDEREÇO: Rua Casa do Ator, 642 - Vila Olímpia, São Paulo\n"
        f"\n"
        f"✅ CONFIRMAÇÃO DE PRESENÇA:\n"
        f"Para nos ajudar com os preparativos, pedimos a gentileza de confirmar sua presença pelo link abaixo:\n"
        f"👉 https://noivos.casar.com/jaqueline-e-willian-2026-07-25#/rsvp\n"
        f"\n"
        f"⚠️ DICA IMPORTANTE: Beba com moderação, mas se não moderar, não dirija! 😂 "
        f"Recomendamos o uso de aplicativos de transporte para que todos voltem em segurança. 🚗💨\n"
        f"🅿️ PARA QUEM FIZER QUESTÃO DE IR DE CARRO: Informamos que o local não possui serviço de valet na porta, "
        f"mas deixamos as opções de estacionamentos mais próximos no nosso site para te ajudar.\n"
        f"\n"
        f"🎁 LISTA DE PRESENTES E MENSAGENS: Nossa 'Lista de Presentes' e a seção de 'Recados' estão disponíveis "
        f"no link oficial. Vamos amar ler cada palavra de carinho de vocês! 😍\n"
        f"🔗 https://noivos.casar.com/jaqueline-e-willian-2026-07-25#/home\n"
        f"\n"
        f"Contamos com a presença de vocês para tornar este dia ainda mais especial e cheio de energia! ✨🙌\n"
        f"\n"
        f"Mal podemos esperar pelo nosso grande dia! 💍\n"
        f"\n"
        f"Will e Jaque"
    )


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
        f.write(f'"{telefone}","{nome}","{status}","{erro}","{timestamp}"\n')


def enviar_convites(modo_teste=False):
    """
    Envia convites para todos os convidados do CSV.

    Args:
        modo_teste: Se True, apenas mostra as mensagens sem enviar.
    """
    # Validar que o PDF existe
    if not modo_teste and not os.path.exists(PDF_CONVITE):
        print(f"❌ ERRO: PDF do convite não encontrado!")
        print(f"   Esperado em: {PDF_CONVITE}")
        print(f"   Coloque o PDF na pasta wedding_invites/ e atualize o nome em config.py")
        return

    with open(ARQUIVO_CSV, "r", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        convidados = [
            (linha["telefone"].strip(), linha["nome"].strip())
            for linha in leitor
            if linha["telefone"].strip() and linha["nome"].strip()
        ]

    total = len(convidados)
    print(f"\n{'='*55}")
    print(f"  💒 ENVIO DE CONVITES - Will e Jaque")
    print(f"  📋 Total de convidados: {total}")
    print(f"  ⏱️  Intervalo entre envios: {INTERVALO_ENVIO_SEGUNDOS}s")
    if not modo_teste:
        print(f"  📄 PDF: {PDF_CONVITE}")
    print(f"{'='*55}\n")

    if modo_teste:
        print("⚠️  MODO TESTE - Nenhuma mensagem será enviada\n")

    enviados = 0
    erros = 0

    for i, (telefone, nome) in enumerate(convidados, 1):
        tel_formatado = formatar_telefone(telefone)
        mensagem = montar_mensagem(nome)

        print(f"[{i}/{total}] {nome} ({tel_formatado})")

        if modo_teste:
            print(f"  📝 Mensagem (início): {mensagem[:80]}...")
            print(f"  ✅ [TESTE] OK\n")
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
            print(f"  ✅ Mensagem enviada!")

            # Aguarda antes de enviar o PDF
            time.sleep(5)

            # Envia o PDF do convite
            kit.sendwhats_image(
                receiver=tel_formatado,
                img_path=PDF_CONVITE,
                caption="Convite de Casamento 💒💍",
                wait_time=15,
                tab_close=True,
            )
            print(f"  ✅ PDF enviado!")

            registrar_log(tel_formatado, nome, "ENVIADO")
            enviados += 1

        except Exception as e:
            print(f"  ❌ ERRO: {e}")
            registrar_log(tel_formatado, nome, "ERRO", str(e))
            erros += 1

        # Intervalo entre envios
        if i < total:
            print(f"  ⏳ Aguardando {INTERVALO_ENVIO_SEGUNDOS}s...\n")
            time.sleep(INTERVALO_ENVIO_SEGUNDOS)

    print(f"\n{'='*55}")
    print(f"  📊 RESULTADO FINAL")
    print(f"  ✅ Enviados: {enviados}/{total}")
    if erros:
        print(f"  ❌ Erros: {erros}")
    print(f"  📄 Log salvo em: {LOG_ENVIO}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import sys

    if "--teste" in sys.argv:
        enviar_convites(modo_teste=True)
    else:
        print("\n⚠️  ATENÇÃO: Este script vai enviar mensagens reais no WhatsApp!")
        print("   Certifique-se de que:")
        print("   1. Você está logado no WhatsApp Web no navegador")
        print("   2. O arquivo convidados.csv está preenchido corretamente")
        print(f"   3. O PDF do convite existe: {PDF_CONVITE}")
        print()
        resposta = input("Deseja continuar? (sim/nao): ").strip().lower()
        if resposta in ("sim", "s", "yes", "y"):
            enviar_convites(modo_teste=False)
        else:
            print("Envio cancelado.")
