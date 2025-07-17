#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para interface de Feature Engineering
Página dedicada à visualização e download dos dados processados para Deep Learning
"""

import streamlit as st
import pandas as pd
from services.data_manager import get_feature_engineering_summary

def create_feature_engineering_interface(df_vagas_dl, df_prospects_dl, df_candidates_dl):
    """
    Cria interface para Feature Engineering
    
    Args:
        df_vagas_dl: DataFrame de vagas processado para Deep Learning
        df_prospects_dl: DataFrame de prospects processado para Deep Learning
        df_candidates_dl: DataFrame de candidatos processado para Deep Learning
    """
    
    st.header("🤖 Feature Engineering para Deep Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Resumo das Features")
        try:
            summary = get_feature_engineering_summary()
            
            for dataset_name, info in summary.items():
                st.write(f"**{dataset_name.title()}:**")
                st.write(f"- Total de features: {info['total_features']}")
                st.write(f"- Encoders utilizados: {info['encoders_used']}")
                st.write(f"- Scalers utilizados: {info['scalers_used']}")
                st.write(f"- Vectorizers utilizados: {info['vectorizers_used']}")
                st.write("---")
        except Exception as e:
            st.error(f"Erro ao gerar resumo: {e}")
    
    with col2:
        st.subheader("📈 Dados Processados")
        
        # Tabs para diferentes datasets
        tab1, tab2, tab3 = st.tabs(["Vagas", "Prospects", "Candidatos"])
        
        with tab1:
            st.write("**Dados de Vagas para Deep Learning:**")
            if not df_vagas_dl.empty:
                st.write(f"Shape: {df_vagas_dl.shape}")
                st.dataframe(df_vagas_dl.head())
                
                if st.button("📥 Download Vagas DL", key="download_vagas"):
                    csv = df_vagas_dl.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="vagas_deep_learning.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Nenhum dado processado disponível")
        
        with tab2:
            st.write("**Dados de Prospects para Deep Learning:**")
            if not df_prospects_dl.empty:
                st.write(f"Shape: {df_prospects_dl.shape}")
                st.dataframe(df_prospects_dl.head())
                
                if st.button("📥 Download Prospects DL", key="download_prospects"):
                    csv = df_prospects_dl.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="prospects_deep_learning.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Nenhum dado processado disponível")
        
        with tab3:
            st.write("**Dados de Candidatos para Deep Learning:**")
            if not df_candidates_dl.empty:
                st.write(f"Shape: {df_candidates_dl.shape}")
                st.dataframe(df_candidates_dl.head())
                
                if st.button("📥 Download Candidatos DL", key="download_candidates"):
                    csv = df_candidates_dl.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="candidatos_deep_learning.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Nenhum dado processado disponível")
    
    # Seção de informações técnicas
    st.header("ℹ️ Informações Técnicas")
    
    with st.expander("🔧 Processamentos Aplicados"):
        st.markdown("""
        **Vagas:**
        - Encoding categórico (One-Hot/Label Encoding)
        - Normalização de valores monetários
        - TF-IDF para títulos de vagas
        - Features derivadas de datas
        - Normalização com StandardScaler
        
        **Prospects:**
        - Encoding de situação e recrutador
        - Extração de informações salariais dos comentários
        - TF-IDF para comentários
        - Features temporais (tempo até atualização)
        - Normalização de datas
        
        **Candidatos:**
        - Encoding de níveis acadêmicos e idiomas
        - Normalização de remuneração
        - TF-IDF para objetivos profissionais e conhecimentos
        - Features derivadas de contato (email, telefone)
        - Normalização temporal
        """)
    
    with st.expander("🎯 Pronto para Deep Learning"):
        st.markdown("""
        **Características dos dados processados:**
        - Todos os valores categóricos foram codificados numericamente
        - Valores numéricos foram normalizados (média 0, desvio padrão 1)
        - Textos foram convertidos em features TF-IDF
        - Datas foram transformadas em features numéricas
        - Valores ausentes foram tratados adequadamente
        - Dados estão prontos para alimentar modelos de Deep Learning
        """)
    
    # Seção de estatísticas detalhadas
    st.header("📊 Estatísticas Detalhadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not df_vagas_dl.empty:
            st.subheader("Vagas")
            st.metric("Total de registros", len(df_vagas_dl))
            st.metric("Total de features", df_vagas_dl.shape[1])
            
            # Estatísticas básicas dos dados numéricos
            numeric_cols = df_vagas_dl.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.metric("Features numéricas", len(numeric_cols))
                st.metric("Valores ausentes", df_vagas_dl.isnull().sum().sum())
    
    with col2:
        if not df_prospects_dl.empty:
            st.subheader("Prospects")
            st.metric("Total de registros", len(df_prospects_dl))
            st.metric("Total de features", df_prospects_dl.shape[1])
            
            numeric_cols = df_prospects_dl.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.metric("Features numéricas", len(numeric_cols))
                st.metric("Valores ausentes", df_prospects_dl.isnull().sum().sum())
    
    with col3:
        if not df_candidates_dl.empty:
            st.subheader("Candidatos")
            st.metric("Total de registros", len(df_candidates_dl))
            st.metric("Total de features", df_candidates_dl.shape[1])
            
            numeric_cols = df_candidates_dl.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.metric("Features numéricas", len(numeric_cols))
                st.metric("Valores ausentes", df_candidates_dl.isnull().sum().sum())
    
