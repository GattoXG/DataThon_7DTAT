import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="DataThon DTAT - Análise de Recrutamento",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports lazy para otimizar carregamento
@st.cache_resource
def get_data_manager():
    """Importa e retorna o data_manager com cache de recurso"""
    from services.data_manager import (
        load_data, process_vagas_data, process_prospects_data, process_applicants_data,
        process_vagas_for_deep_learning, process_prospects_for_deep_learning, 
        process_candidates_for_deep_learning
    )
    return {
        'load_data': load_data,
        'process_vagas_data': process_vagas_data,
        'process_prospects_data': process_prospects_data,
        'process_applicants_data': process_applicants_data,
        'process_vagas_for_deep_learning': process_vagas_for_deep_learning,
        'process_prospects_for_deep_learning': process_prospects_for_deep_learning,
        'process_candidates_for_deep_learning': process_candidates_for_deep_learning
    }

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_and_process_basic_data():
    """Carrega e processa dados básicos com cache otimizado"""
    dm = get_data_manager()
    
    vagas, prospects, applicants = dm['load_data']()
    
    if not vagas or not prospects or not applicants:
        return None, None, None, None, None, None
    
    # Processamento básico
    df_vagas = dm['process_vagas_data'](vagas)
    df_prospects = dm['process_prospects_data'](prospects)
    df_candidates = dm['process_applicants_data'](applicants)
    
    return vagas, prospects, applicants, df_vagas, df_prospects, df_candidates

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_and_process_dl_data():
    """Carrega e processa dados para Deep Learning com cache otimizado"""
    dm = get_data_manager()
    
    vagas, prospects, applicants = dm['load_data']()
    
    if not vagas or not prospects or not applicants:
        return None, None, None
    
    # Processamento para Deep Learning (apenas quando necessário)
    df_vagas_dl = dm['process_vagas_for_deep_learning'](vagas)
    df_prospects_dl = dm['process_prospects_for_deep_learning'](prospects)
    df_candidates_dl = dm['process_candidates_for_deep_learning'](applicants)
    
    return df_vagas_dl, df_prospects_dl, df_candidates_dl

def main():
    # Título principal
    st.title("📊 DataThon DTAT - Sistema de Análise de Recrutamento")
    st.markdown("---")
    
    # Mostra indicador de carregamento
    with st.spinner("Carregando dados..."):
        # Carrega dados básicos (com cache)
        vagas, prospects, applicants, df_vagas, df_prospects, df_candidates = load_and_process_basic_data()
    
    if vagas is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos JSON estão no diretório correto.")
        return
    
    # Barra lateral de navegação
    st.sidebar.header("🧭 Navegação")
    
    # Inicializa o estado da sessão se não existir
    if 'pagina_atual' not in st.session_state:
        st.session_state.pagina_atual = "📈 Vagas"
    
    # Botões de navegação com melhor layout
    paginas = [
        ("📈 Vagas", "📈 Vagas"),
        ("👤 Candidatos", "👤 Candidatos"),
        ("🔬 Análises Avançadas", "🔬 Análises Avançadas"),
        ("🤖 Feature Engineering", "🤖 Feature Engineering"),
        ("🧠 Deep Learning", "🧠 Deep Learning"),
        ("🎯 Job Matching", "🎯 Job Matching")
    ]
    
    for label, key in paginas:
        if st.sidebar.button(label, use_container_width=True, key=f"btn_{key}"):
            st.session_state.pagina_atual = key
    
    # Métricas principais na sidebar (carregamento lazy)
    with st.sidebar.expander("📊 Métricas Principais", expanded=True):
        from pages.metrics_dashboard import create_metrics_dashboard
        create_metrics_dashboard(df_vagas, df_prospects, df_candidates)
    
    # Conteúdo baseado na seleção
    if st.session_state.pagina_atual == "📈 Vagas":
        # Análise de Vagas
        from pages.vagas_analysis import create_vagas_analysis_interface
        create_vagas_analysis_interface(df_vagas)
    
    elif st.session_state.pagina_atual == "👤 Candidatos":
        # Análise de candidatos
        from pages.candidates_analysis import show_applicants_analysis
        show_applicants_analysis(df_candidates)
    
    elif st.session_state.pagina_atual == "🔬 Análises Avançadas":
        # Análises avançadas
        from pages.advanced_analysis import create_advanced_analysis
        create_advanced_analysis(df_vagas, df_prospects)
    
    elif st.session_state.pagina_atual == "🤖 Feature Engineering":
        # Feature Engineering - carrega dados DL apenas quando necessário
        df_vagas_dl, df_prospects_dl, df_candidates_dl = load_and_process_dl_data()
        if df_vagas_dl is not None:
            from pages.feature_engineering_analysis import create_feature_engineering_interface
            create_feature_engineering_interface(df_vagas_dl, df_prospects_dl, df_candidates_dl)
        else:
            st.error("Erro ao carregar dados para Feature Engineering.")
    
    elif st.session_state.pagina_atual == "🧠 Deep Learning":
        # Deep Learning - carrega dados DL apenas quando necessário
        df_vagas_dl, df_prospects_dl, df_candidates_dl = load_and_process_dl_data()
        if df_vagas_dl is not None:
            from pages.deep_learning_interface import create_deep_learning_interface
            create_deep_learning_interface(df_vagas_dl, df_prospects_dl, df_candidates_dl)
        else:
            st.error("Erro ao carregar dados para Deep Learning.")
        
    elif st.session_state.pagina_atual == "🎯 Job Matching":
        # Job Matching
        from pages.job_matching_interface import create_job_matching_interface
        create_job_matching_interface()

if __name__ == "__main__":
    main()
