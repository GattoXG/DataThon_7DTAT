import streamlit as st

def create_metrics_dashboard(df_vagas, df_prospects, df_applicants):
    """Interface para métricas principais do dashboard na sidebar"""
    # Métricas principais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Métricas Gerais")
    
    st.sidebar.metric("Total de Vagas", len(df_vagas))
    st.sidebar.metric("Total de Prospects", len(df_prospects))
    st.sidebar.metric("Total de Candidatos", len(df_applicants))
