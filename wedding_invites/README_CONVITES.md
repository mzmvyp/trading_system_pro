# 💒 Convites de Casamento - Will e Jaque

## Como usar

### 1. Instalar dependências
```bash
pip install -r wedding_invites/requirements.txt
```

### 2. Preparar os arquivos
- **`convidados.csv`** - Preencha com telefone e nome de cada convidado
- **`convite_will_jaque.pdf`** - Coloque o PDF da arte do convite nesta pasta
- **`config.py`** - Ajuste o nome do PDF e o intervalo entre envios se necessário

### 3. Formato do CSV
```
telefone,nome
5511999990001,Priscila e Lucas
5511999990002,Carlos
5511999990003,Família Silva
```

### 4. Testar (sem enviar nada)
```bash
cd wedding_invites
python enviar_whatsapp.py --teste
```

### 5. Enviar de verdade
```bash
python enviar_whatsapp.py
```

### ⚠️ Importante
- Faça login no **WhatsApp Web** no navegador antes de enviar
- Não mexa no computador durante os envios
- Intervalo de 45s entre envios (configurável em `config.py`)
- Log de envios salvo em `log_envios.csv`
