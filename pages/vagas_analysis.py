import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

def create_vagas_analysis_interface(df_vagas):
    """Interface para análise de vagas"""
    st.header("📈 Análise de Vagas")
    
    # Tabs para organizar melhor o conteúdo
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "🗺️ Análise Geográfica", "💼 Análise SAP", "💰 Análise Salarial"])
    
    with tab1:
        st.subheader("📊 Visão Geral das Vagas")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Vagas", len(df_vagas))
        
        with col2:
            vagas_com_salario = len(df_vagas[df_vagas['valor_venda_num'].notna() & (df_vagas['valor_venda_num'] > 0)])
            st.metric("Vagas c/ Salário", vagas_com_salario)
        
        with col3:
            vagas_sap = len(df_vagas[df_vagas['vaga_sap'] == 'Sim'])
            st.metric("Vagas SAP", vagas_sap)
        
        with col4:
            salario_medio = df_vagas[df_vagas['valor_venda_num'] > 0]['valor_venda_num'].mean()
            st.metric("Salário Médio", f"R$ {salario_medio:,.0f}" if not pd.isna(salario_medio) else "N/A")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de níveis profissionais
            nivel_prof = df_vagas['nivel_profissional'].value_counts()
            fig_nivel = px.bar(
                x=nivel_prof.values,
                y=nivel_prof.index,
                orientation='h',
                title="📊 Distribuição por Nível Profissional",
                color=nivel_prof.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_nivel, use_container_width=True)
        
        with col2:
            # Gráfico de áreas de atuação
            areas_atuacao = df_vagas['areas_atuacao'].value_counts().head(10)
            fig_areas = px.bar(
                x=areas_atuacao.values,
                y=areas_atuacao.index,
                orientation='h',
                title="🎯 Top 10 Áreas de Atuação",
                color=areas_atuacao.values,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_areas, use_container_width=True)
    
    with tab2:
        st.subheader("🗺️ Análise Geográfica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de vagas por estado
            vagas_por_estado = df_vagas['estado'].value_counts()
            fig_estados = px.bar(
                x=vagas_por_estado.index,
                y=vagas_por_estado.values,
                title="📍 Distribuição de Vagas por Estado",
                labels={'x': 'Estado', 'y': 'Número de Vagas'},
                color=vagas_por_estado.values,
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_estados, use_container_width=True)
        
        with col2:
            # Gráfico de vagas por cidade (top 15)
            vagas_por_cidade = df_vagas['cidade'].value_counts().head(15)
            fig_cidades = px.bar(
                x=vagas_por_cidade.values,
                y=vagas_por_cidade.index,
                orientation='h',
                title="🏙️ Top 15 Cidades com Mais Vagas",
                color=vagas_por_cidade.values,
                color_continuous_scale='greens'
            )
            st.plotly_chart(fig_cidades, use_container_width=True)
        
        # Análise combinada Estado-Cidade
        st.subheader("📊 Análise Estado-Cidade")
        estado_cidade = df_vagas.groupby(['estado', 'cidade']).size().reset_index(name='count')
        estado_cidade = estado_cidade.sort_values('count', ascending=False).head(20)
        
        fig_combinado = px.bar(
            estado_cidade,
            x='count',
            y='cidade',
            color='estado',
            orientation='h',
            title="🏢 Top 20 Combinações Estado-Cidade",
            hover_data=['estado']
        )
        st.plotly_chart(fig_combinado, use_container_width=True)
    
    with tab3:
        st.subheader("💼 Análise de Vagas SAP")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de vagas SAP vs Não SAP
            vagas_sap = df_vagas['vaga_sap'].value_counts()
            fig_sap = px.pie(
                values=vagas_sap.values,
                names=vagas_sap.index,
                title="💼 Distribuição: Vagas SAP vs Não SAP",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4']
            )
            st.plotly_chart(fig_sap, use_container_width=True)
        
        with col2:
            # Análise SAP por nível profissional
            sap_nivel = pd.crosstab(df_vagas['nivel_profissional'], df_vagas['vaga_sap'])
            fig_sap_nivel = px.bar(
                x=sap_nivel.index,
                y=sap_nivel['Sim'],
                title="📊 Vagas SAP por Nível Profissional",
                color=sap_nivel['Sim'],
                color_continuous_scale='oranges'
            )
            st.plotly_chart(fig_sap_nivel, use_container_width=True)
        
        # Análise SAP por estado
        st.subheader("🗺️ Distribuição SAP por Estado")
        sap_estado = pd.crosstab(df_vagas['estado'], df_vagas['vaga_sap'])
        sap_estado['Total'] = sap_estado.sum(axis=1)
        sap_estado['Percentual_SAP'] = (sap_estado['Sim'] / sap_estado['Total'] * 100).round(1)
        sap_estado = sap_estado.sort_values('Percentual_SAP', ascending=False)
        
        fig_sap_estado = px.bar(
            x=sap_estado.index,
            y=sap_estado['Percentual_SAP'],
            title="📈 Percentual de Vagas SAP por Estado",
            labels={'y': 'Percentual SAP (%)', 'x': 'Estado'},
            color=sap_estado['Percentual_SAP'],
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig_sap_estado, use_container_width=True)
    
    with tab4:
        st.subheader("💰 Análise Salarial")
        
        # Filtrar apenas vagas com salário informado
        df_com_salario = df_vagas[df_vagas['valor_venda_num'].notna() & (df_vagas['valor_venda_num'] > 0)]
        
        if not df_com_salario.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição salarial
                fig_dist_salario = px.histogram(
                    df_com_salario,
                    x='valor_venda_num',
                    nbins=20,
                    title="📊 Distribuição Salarial",
                    labels={'x': 'Salário (R$)', 'y': 'Quantidade de Vagas'}
                )
                st.plotly_chart(fig_dist_salario, use_container_width=True)
            
            with col2:
                # Box plot por nível profissional
                fig_box_nivel = px.box(
                    df_com_salario,
                    x='nivel_profissional',
                    y='valor_venda_num',
                    title="📈 Salários por Nível Profissional"
                )
                fig_box_nivel.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box_nivel, use_container_width=True)
            
            # Análise salarial por estado
            st.subheader("🗺️ Análise Salarial por Estado")
            salario_estado = df_com_salario.groupby('estado')['valor_venda_num'].agg(['mean', 'median', 'count']).round(0)
            salario_estado.columns = ['Média', 'Mediana', 'Qtd_Vagas']
            salario_estado = salario_estado[salario_estado['Qtd_Vagas'] >= 3].sort_values('Média', ascending=False)
            
            fig_salario_estado = px.bar(
                x=salario_estado.index,
                y=salario_estado['Média'],
                title="💰 Salário Médio por Estado (min. 3 vagas)",
                labels={'y': 'Salário Médio (R$)', 'x': 'Estado'},
                color=salario_estado['Média'],
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_salario_estado, use_container_width=True)
            
            # Tabela detalhada
            st.subheader("📋 Resumo Salarial por Estado")
            st.dataframe(salario_estado.style.format({'Média': 'R$ {:,.0f}', 'Mediana': 'R$ {:,.0f}'}))
        else:
            pass  # Não há dados suficientes de salário para análise
    
    st.markdown("---")
