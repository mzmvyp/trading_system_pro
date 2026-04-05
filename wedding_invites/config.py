# ============================================================
# CONFIGURAÇÕES DO CONVITE - Will e Jaque
# ============================================================

import os

# Caminho do PDF do convite (o mesmo para todos)
# Coloque o arquivo PDF na pasta wedding_invites/ e atualize o nome abaixo
PDF_CONVITE = os.path.join(os.path.dirname(__file__), "convite_will_jaque.pdf")

# Intervalo entre envios no WhatsApp (em segundos)
# Mínimo recomendado: 30 segundos para evitar bloqueio
INTERVALO_ENVIO_SEGUNDOS = 45
