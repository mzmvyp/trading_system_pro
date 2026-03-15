#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de inicialização do Sistema de Trading
"""

import subprocess
import sys
import os
import psutil
import signal
import time

def kill_all_python_processes():
    """Mata todos os processos Python relacionados ao sistema de trading"""
    print("Verificando e matando processos antigos...")
    
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Verifica se é um processo Python
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline:
                    cmdline_str = ' '.join(cmdline)
                    
                    # Verifica se é um processo do sistema de trading
                    trading_keywords = [
                        'start_data_collection.py',
                        'start_trading_system.py', 
                        'run_data_stream.py',
                        'run_system.py',
                        'streamlit_dashboard.py',
                        'main.py',
                        'binance_data_collector.py',
                        'backtest_engine.py',
                        'optimization_engine.py'
                    ]
                    
                    if any(keyword in cmdline_str for keyword in trading_keywords):
                        print(f"   Matando processo: {proc.info['pid']} - {cmdline_str[:80]}...")
                        try:
                            proc.kill()
                            killed_count += 1
                            time.sleep(0.1)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        print(f"   {killed_count} processos antigos finalizados")
        time.sleep(2)  # Aguarda processos terminarem
    else:
        print("   Nenhum processo antigo encontrado")

def kill_streamlit_processes():
    """Mata processos Streamlit específicos"""
    print("Verificando processos Streamlit...")
    
    killed_count = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'streamlit' in proc.info['name'].lower():
                print(f"   Matando Streamlit: {proc.info['pid']}")
                try:
                    proc.kill()
                    killed_count += 1
                    time.sleep(0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed_count > 0:
        print(f"   {killed_count} processos Streamlit finalizados")
        time.sleep(2)

def main():
    print("Sistema de Trading - Inicializacao LIMPA")
    print("=" * 50)
    
    # 1. Mata todos os processos antigos
    kill_all_python_processes()
    kill_streamlit_processes()
    
    print("\n1. Iniciando coleta de dados...")
    try:
        # Inicia coleta de dados em background
        data_process = subprocess.Popen([sys.executable, "binance_data_collector.py"])
        print("   Coleta de dados iniciada em background")
    except Exception as e:
        print(f"Erro na coleta de dados: {e}")
        return
    
    print("\n2. Iniciando dashboard...")
    try:
        # Inicia dashboard em background
        dashboard_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_dashboard.py"])
        print("   Dashboard iniciado em background")
        print("\n3. Sistema iniciado com sucesso!")
        print("   - Coleta de dados: rodando")
        print("   - Dashboard: rodando")
        print("   - Acesse: http://localhost:8501")
        
        # Aguarda um pouco para verificar se os processos iniciaram
        time.sleep(3)
        
        # Verifica se os processos ainda estão rodando
        if data_process.poll() is None:
            print("   OK - Coleta de dados: funcionando")
        else:
            print("   ERRO - Coleta de dados: falhou")
            
        if dashboard_process.poll() is None:
            print("   OK - Dashboard: funcionando")
        else:
            print("   ERRO - Dashboard: falhou")
            
    except Exception as e:
        print(f"Erro ao iniciar dashboard: {e}")
        print("Execute manualmente: streamlit run dashboard/streamlit_dashboard.py")

if __name__ == "__main__":
    main()
