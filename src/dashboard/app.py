"""
Entry point do dashboard Streamlit.
Redireciona para o dashboard principal na raiz (streamlit_dashboard.py).
Para rodar: streamlit run streamlit_dashboard.py
Ou: streamlit run dashboard/app.py (se este chamar o outro).
"""
# O dashboard principal está em streamlit_dashboard.py na raiz.
# Esta estrutura (dashboard/pages/, components/) está pronta para migração futura
# para multi-páginas: overview, backtesting, live_trading, ml_models, signals.
import sys
from pathlib import Path

# Adiciona raiz do projeto
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Executa o dashboard principal (agora em src/dashboard/)
import subprocess
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve().parent / "streamlit_app.py"),
        "--server.headless", "true"
    ], cwd=str(root))
