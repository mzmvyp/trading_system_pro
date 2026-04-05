# 💒 Sistema de Convites de Casamento via WhatsApp

## Como usar

### 1. Instalar dependências
```bash
pip install -r wedding_invites/requirements.txt
```

### 2. Editar suas informações
- **`config.py`** - Dados do casamento (nomes, data, local, cores, etc.)
- **`convidados.csv`** - Lista de convidados (telefone + nome)

### 3. Formato do CSV
```
telefone,nome
5511999990001,João e Maria Silva
5511999990002,Pedro e Ana Santos
```
- Telefone: com DDD, sem espaços (ex: `5511999990001`)
- Nome: nome completo ou do casal

### 4. Gerar os PDFs
```bash
cd wedding_invites
python gerar_pdf.py
```
Os PDFs ficam na pasta `convites_pdf/`.

### 5. Enviar pelo WhatsApp

**Modo teste** (não envia nada, só mostra):
```bash
python enviar_whatsapp.py --teste
```

**Envio real:**
```bash
python enviar_whatsapp.py
```

### ⚠️ Importante
- Faça login no **WhatsApp Web** no navegador antes de enviar
- Não mexa no computador durante os envios
- O intervalo entre envios é de 45s (configurável em `config.py`)
- Um log de envios é salvo em `log_envios.csv`
