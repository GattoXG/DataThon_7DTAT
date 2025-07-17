import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

def show_applicants_analysis(df_candidates):
    """Exibe análise detalhada dos candidatos"""
    st.header("👥 Análise Detalhada de Candidatos")
    
    # Tabs para melhor organização
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Perfil Geral", "🎓 Formação & Idiomas", "💰 Remuneração", "🔍 Análise Detalhada"])
    
    with tab1:
        st.subheader("📊 Perfil Geral dos Candidatos")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Candidatos", len(df_candidates))
        
        with col2:
            nivel_superior = len(df_candidates[df_candidates['nivel_academico'].str.contains('Superior|MBA|Pós', na=False, case=False)])
            st.metric("Nível Superior+", nivel_superior)
        
        with col3:
            ingles_avancado = len(df_candidates[df_candidates['nivel_ingles'].str.contains('Avançado|Fluente', na=False, case=False)])
            st.metric("Inglês Avançado/Fluente", ingles_avancado)
        
        with col4:
            com_objetivo = len(df_candidates[df_candidates['objetivo_profissional'].str.strip() != ''])
            st.metric("Com Objetivo Definido", com_objetivo)
        
        st.markdown("---")
        
        # Análise de áreas de atuação
        col1, col2 = st.columns(2)
        
        with col1:
            # Top áreas de atuação
            areas_atuacao = df_candidates['area_atuacao'].value_counts().head(15)
            fig_areas = px.bar(
                x=areas_atuacao.values,
                y=areas_atuacao.index,
                orientation='h',
                title="🎯 Top 15 Áreas de Atuação",
                color=areas_atuacao.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_areas, use_container_width=True)
        
        with col2:
            # Distribuição por inserido_por
            inserido_por = df_candidates['inserido_por'].value_counts().head(10)
            fig_inserido = px.bar(
                x=inserido_por.values,
                y=inserido_por.index,
                orientation='h',
                title="👤 Top 10 Responsáveis por Inserção",
                color=inserido_por.values,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_inserido, use_container_width=True)
    
    with tab2:
        st.subheader("🎓 Análise de Formação e Idiomas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por nível acadêmico
            nivel_academico = df_candidates['nivel_academico'].value_counts()
            if not nivel_academico.empty:
                fig_academico = px.bar(
                    x=nivel_academico.values,
                    y=nivel_academico.index,
                    orientation='h',
                    title="📚 Distribuição por Nível Acadêmico",
                    color=nivel_academico.values,
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig_academico, use_container_width=True)
            else:
                st.info("Dados de nível acadêmico não disponíveis")
        
        with col2:
            # Distribuição por nível de inglês
            nivel_ingles = df_candidates['nivel_ingles'].value_counts()
            if not nivel_ingles.empty:
                fig_ingles = px.bar(
                    x=nivel_ingles.values,
                    y=nivel_ingles.index,
                    orientation='h',
                    title="🌍 Distribuição por Nível de Inglês",
                    color=nivel_ingles.values,
                    color_continuous_scale='greens'
                )
                st.plotly_chart(fig_ingles, use_container_width=True)
            else:
                st.info("Dados de nível de inglês não disponíveis")
        
        # Análise de idiomas combinados
        st.subheader("🗣️ Análise Combinada de Idiomas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por nível de espanhol
            nivel_espanhol = df_candidates['nivel_espanhol'].value_counts()
            if not nivel_espanhol.empty:
                fig_espanhol = px.pie(
                    values=nivel_espanhol.values,
                    names=nivel_espanhol.index,
                    title="🇪🇸 Distribuição por Nível de Espanhol"
                )
                st.plotly_chart(fig_espanhol, use_container_width=True)
            else:
                st.info("Dados de nível de espanhol não disponíveis")
        
        with col2:
            # Candidatos bilíngues/multilíngues
            candidatos_bilingues = df_candidates[
                (df_candidates['nivel_ingles'].str.contains('Avançado|Fluente', na=False, case=False)) &
                (df_candidates['nivel_espanhol'].str.contains('Avançado|Fluente', na=False, case=False))
            ]
            
            st.metric("Candidatos Bilíngues", len(candidatos_bilingues))
            st.metric("% Bilíngues", f"{len(candidatos_bilingues)/len(df_candidates)*100:.1f}%")
            
            # Gráfico de qualificação linguística
            qualificacao = pd.DataFrame({
                'Categoria': ['Apenas Português', 'Inglês Avançado', 'Espanhol Avançado', 'Bilíngues'],
                'Quantidade': [
                    len(df_candidates) - len(df_candidates[df_candidates['nivel_ingles'].str.contains('Avançado|Fluente', na=False, case=False)]) - len(df_candidates[df_candidates['nivel_espanhol'].str.contains('Avançado|Fluente', na=False, case=False)]),
                    len(df_candidates[df_candidates['nivel_ingles'].str.contains('Avançado|Fluente', na=False, case=False)]) - len(candidatos_bilingues),
                    len(df_candidates[df_candidates['nivel_espanhol'].str.contains('Avançado|Fluente', na=False, case=False)]) - len(candidatos_bilingues),
                    len(candidatos_bilingues)
                ]
            })
            
            fig_qualificacao = px.bar(
                qualificacao,
                x='Categoria',
                y='Quantidade',
                title="🎯 Qualificação Linguística",
                color='Quantidade',
                color_continuous_scale='oranges'
            )
            st.plotly_chart(fig_qualificacao, use_container_width=True)
    
    with tab3:
        st.subheader("💰 Análise de Remuneração")
        
        if 'remuneracao_num' in df_candidates.columns:    
            
            # Remove outliers extremos mas mais permissivo
            remuneracao_valida = df_candidates.dropna(subset=['remuneracao_num'])
            remuneracao_valida = remuneracao_valida[
                (remuneracao_valida['remuneracao_num'] >= 500) &  # Mais permissivo
                (remuneracao_valida['remuneracao_num'] <= 1000000)  # Mais permissivo para executivos
            ]
            
            if not remuneracao_valida.empty and len(remuneracao_valida) > 1:
                col1, col2 = st.columns(2)
            
            with col1:
                # Histograma de remuneração com faixas customizadas
                # Define faixas menores para melhor visualização
                bins = [0, 1500, 2500, 3500, 4500, 6000, 8000, 12000, 20000, 35000, 500000]
                bin_labels = [
                    'Até R$ 1.5k', 'R$ 1.5k - 2.5k', 'R$ 2.5k - 3.5k', 
                    'R$ 3.5k - 4.5k', 'R$ 4.5k - 6k', 'R$ 6k - 8k',
                    'R$ 8k - 12k', 'R$ 12k - 20k', 'R$ 20k - 35k', 'Acima R$ 35k'
                ]
                
                # Categoriza os salários em faixas
                remuneracao_valida['faixa_salarial'] = pd.cut(
                    remuneracao_valida['remuneracao_num'], 
                    bins=bins, 
                    labels=bin_labels, 
                    include_lowest=True
                )
                
                # Conta por faixa
                faixas_count = remuneracao_valida['faixa_salarial'].value_counts().sort_index()
                
                fig_remuneracao = px.bar(
                    x=faixas_count.index,
                    y=faixas_count.values,
                    title="💰 Distribuição de Remuneração por Faixas",
                    labels={'x': 'Faixa Salarial', 'y': 'Número de Candidatos'}
                )
                fig_remuneracao.update_xaxes(tickangle=45)
                st.plotly_chart(fig_remuneracao, use_container_width=True)
                
                # Gráfico adicional: zoom na faixa mais comum (até R$ 15k)
                faixa_comum = remuneracao_valida[remuneracao_valida['remuneracao_num'] <= 15000]
                if not faixa_comum.empty:
                    fig_zoom = px.histogram(
                        faixa_comum,
                        x='remuneracao_num',
                        title="🔍 Zoom: Distribuição até R$ 15.000",
                        labels={'remuneracao_num': 'Remuneração (R$)', 'count': 'Frequência'},
                        nbins=15
                    )
                    st.plotly_chart(fig_zoom, use_container_width=True)
            
            with col2:
                # Estatísticas de remuneração
                st.markdown("**📊 Estatísticas de Remuneração**")
                media = remuneracao_valida['remuneracao_num'].mean()
                mediana = remuneracao_valida['remuneracao_num'].median()
                minimo = remuneracao_valida['remuneracao_num'].min()
                maximo = remuneracao_valida['remuneracao_num'].max()
                
                col_stat1, col_stat2 = st.columns(2)
                with col_stat1:
                    st.metric("Média", f"R$ {media:,.0f}")
                    st.metric("Mediana", f"R$ {mediana:,.0f}")
                with col_stat2:
                    st.metric("Mínimo", f"R$ {minimo:,.0f}")
                    st.metric("Máximo", f"R$ {maximo:,.0f}")
                
                # Tabela de distribuição por faixas
                st.markdown("**📊 Distribuição por Faixas**")
                faixas_df = pd.DataFrame({
                    'Faixa Salarial': faixas_count.index,
                    'Quantidade': faixas_count.values,
                    'Percentual': (faixas_count.values / faixas_count.sum() * 100).round(1)
                })
                faixas_df['Percentual'] = faixas_df['Percentual'].astype(str) + '%'
                st.dataframe(faixas_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Dados de remuneração insuficientes após limpeza e filtros")
    
    st.markdown("---")
    
    # === SEÇÃO 3: PERFIL PROFISSIONAL ===
    st.subheader("🎯 Perfil Profissional")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top áreas de atuação
        areas_atuacao = df_candidates['area_atuacao'].value_counts().head(10)
        if not areas_atuacao.empty:
            fig_areas = px.bar(
                x=areas_atuacao.values,
                y=areas_atuacao.index,
                orientation='h',
                title="🎯 Top 10 Áreas de Atuação"
            )
            st.plotly_chart(fig_areas, use_container_width=True)
    
    with col2:
        # Top responsáveis por inserção
        top_inseridores = df_candidates['inserido_por'].value_counts().head(10)
        if not top_inseridores.empty:
            fig_inseridores = px.bar(
                x=top_inseridores.index,
                y=top_inseridores.values,
                title="👤 Top 10 Responsáveis por Inserção"
            )
            fig_inseridores.update_xaxes(tickangle=45)
            st.plotly_chart(fig_inseridores, use_container_width=True)
    
    st.markdown("---")
    
    # === SEÇÃO 4: DADOS DETALHADOS ===
    st.subheader("📋 Dados Detalhados dos Candidatos")
    
    # Filtros para a tabela
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nivel_filter = st.selectbox(
            "Filtrar por Nível Acadêmico:",
            options=['Todos'] + list(df_candidates['nivel_academico'].dropna().unique()),
            index=0
        )
    
    with col2:
        ingles_filter = st.selectbox(
            "Filtrar por Inglês:",
            options=['Todos'] + list(df_candidates['nivel_ingles'].dropna().unique()),
            index=0
        )
    
    with col3:
        area_filter = st.selectbox(
            "Filtrar por Área:",
            options=['Todas'] + list(df_candidates['area_atuacao'].dropna().unique()),
            index=0
        )
    
    with col4:
        # Filtro por faixa salarial
        salario_filter = st.selectbox(
            "Filtrar por Faixa Salarial:",
            options=['Todas', 'Até R$ 2.500', 'R$ 2.500 - 5.000', 'R$ 5.000 - 10.000', 'R$ 10.000 - 20.000', 'Acima R$ 20.000', 'Sem salário informado'],
            index=0
        )
    
    # Aplicar filtros
    df_filtered = df_candidates.copy()
    
    if nivel_filter != 'Todos':
        df_filtered = df_filtered[df_filtered['nivel_academico'] == nivel_filter]
    
    if ingles_filter != 'Todos':
        df_filtered = df_filtered[df_filtered['nivel_ingles'] == ingles_filter]
    
    if area_filter != 'Todas':
        df_filtered = df_filtered[df_filtered['area_atuacao'] == area_filter]
    
    # Filtro por faixa salarial
    if salario_filter != 'Todas':
        if salario_filter == 'Até R$ 2.500':
            df_filtered = df_filtered[df_filtered['remuneracao_num'] <= 2500]
        elif salario_filter == 'R$ 2.500 - 5.000':
            df_filtered = df_filtered[(df_filtered['remuneracao_num'] > 2500) & (df_filtered['remuneracao_num'] <= 5000)]
        elif salario_filter == 'R$ 5.000 - 10.000':
            df_filtered = df_filtered[(df_filtered['remuneracao_num'] > 5000) & (df_filtered['remuneracao_num'] <= 10000)]
        elif salario_filter == 'R$ 10.000 - 20.000':
            df_filtered = df_filtered[(df_filtered['remuneracao_num'] > 10000) & (df_filtered['remuneracao_num'] <= 20000)]
        elif salario_filter == 'Acima R$ 20.000':
            df_filtered = df_filtered[df_filtered['remuneracao_num'] > 20000]
        elif salario_filter == 'Sem salário informado':
            df_filtered = df_filtered[df_filtered['remuneracao_num'].isna()]
    
    # Exibir tabela filtrada
    colunas_exibir = ['nome', 'objetivo_profissional', 'nivel_academico', 'nivel_ingles', 
                      'area_atuacao', 'remuneracao_num']
    
    # Renomeia as colunas para exibição
    df_display = df_filtered[colunas_exibir].copy()
    df_display = df_display.rename(columns={
        'nome': 'Nome',
        'objetivo_profissional': 'Objetivo Profissional',
        'nivel_academico': 'Nível Acadêmico',
        'nivel_ingles': 'Inglês',
        'area_atuacao': 'Área de Atuação',
        'remuneracao_num': 'Remuneração Mensal (R$)'
    })
    
    # Formata a coluna de remuneração normalizada
    if 'Remuneração Mensal (R$)' in df_display.columns:
        df_display['Remuneração Mensal (R$)'] = df_display['Remuneração Mensal (R$)'].apply(
            lambda x: f"R$ {x:,.0f}".replace(',', '.') if pd.notna(x) else ""
        )
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )
