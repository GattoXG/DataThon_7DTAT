#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface para Job Matching
Permite fazer matching entre candidatos e vagas usando modelos treinados
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_job_matching_interface():
    """
    Cria interface para job matching
    """
    
    st.header("🎯 Job Matching - Predição de Contratações")
    st.markdown("Use modelos treinados para encontrar os melhores matches entre candidatos e vagas.")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs([
        "🔧 Configuração do Modelo", 
        "👤 Candidato → Vagas", 
        "💼 Vaga → Candidatos"
    ])
    
    with tab1:
        show_model_selection()
    
    with tab2:
        show_candidate_to_jobs_matching()
    
    with tab3:
        show_job_to_candidates_matching()

def show_model_selection():
    """Mostra seleção e carregamento do modelo"""
    st.subheader("🔧 Seleção do Modelo")
    
    # Importa o serviço
    from services.job_matching_service import JobMatchingService
    
    # Inicializa serviço se não existir
    if 'job_matching_service' not in st.session_state:
        st.session_state.job_matching_service = JobMatchingService()
    
    service = st.session_state.job_matching_service
    
    # Lista modelos disponíveis
    available_models = service.get_available_models()
    
    if not available_models:
        st.warning("⚠️ Nenhum modelo treinado encontrado. Treine um modelo primeiro na aba 'Deep Learning'.")
        return
    
    # Seleção do modelo
    model_names = [model['display_name'] for model in available_models]
    
    selected_idx = st.selectbox(
        "Selecione o modelo para usar no matching:",
        range(len(model_names)),
        format_func=lambda x: model_names[x],
        key="selected_model_idx"
    )
    
    selected_model = available_models[selected_idx]
    
    # Carrega modelo
    if st.button("🔄 Carregar Modelo", type="primary"):
        with st.spinner("Carregando modelo..."):
            try:
                model_info = service.load_model(selected_model['filepath'])
                st.session_state.model_loaded = True
                st.session_state.current_model_info = model_info
            except Exception as e:
                st.error(f"❌ Erro ao carregar modelo: {str(e)}")
    
    # Mostra informações do modelo carregado
    if hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
        # Modelo carregado e pronto para uso
        
        # Mostra informações do modelo se disponível
        if hasattr(st.session_state, 'current_model_info') and st.session_state.current_model_info:
            eval_results = st.session_state.current_model_info.get('evaluation_results', {})
            if eval_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia", f"{eval_results.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precisão", f"{eval_results.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{eval_results.get('recall', 0):.3f}")
        
        st.info("🎯 Agora você pode usar as abas 'Candidato → Vagas' e 'Vaga → Candidatos' para fazer matching.")

def show_candidate_to_jobs_matching():
    """Mostra interface para encontrar vagas para um candidato"""
    st.subheader("👤 Encontrar Vagas para Candidato")
    
    # Verifica se modelo está carregado
    if not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded):
        st.warning("⚠️ Carregue um modelo primeiro na aba 'Configuração do Modelo'.")
        return
    
    service = st.session_state.job_matching_service
    
    # Carrega dados de candidatos
    try:
        candidates_df = service.get_available_candidates()
        if candidates_df.empty:
            st.warning("⚠️ Nenhum candidato não contratado encontrado.")
            return
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar candidatos: {str(e)}")
        return
    
    # Seleção do candidato
    candidate_options = []
    for idx, row in candidates_df.iterrows():
        candidate_id = row.get('candidato_id', str(idx))
        nome = row.get('nome', f'Candidato {candidate_id}')
        senioridade = row.get('senioridade', 'N/A')
        area = row.get('area', 'N/A')
        candidate_options.append({
            'id': candidate_id,
            'display': f"{nome} - {senioridade} - {area}",
            'info': row
        })
    
    selected_candidate_idx = st.selectbox(
        "Selecione o candidato:",
        range(len(candidate_options)),
        format_func=lambda x: candidate_options[x]['display'],
        key="selected_candidate"
    )
    
    selected_candidate = candidate_options[selected_candidate_idx]
    
    # Mostra informações do candidato
    st.info(f"📋 **Candidato Selecionado:** {selected_candidate['display']}")
    
    # Parâmetros de busca
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Número de vagas a retornar:", 5, 50, 15, key="candidate_to_jobs_top_n")
    
    with col2:
        min_probability = st.slider("Probabilidade mínima:", 0.0, 1.0, 0.1, 0.01, key="candidate_to_jobs_min_prob")
    
    with col3:
        st.info("🤖 Sistema usa modelo treinado para predições reais")
    
    # Botão para buscar vagas
    if st.button("🔍 Buscar Melhores Vagas", type="primary", key="search_jobs_btn"):
        with st.spinner("Analisando matches com modelo de Deep Learning..."):
            try:
                matches = service.find_best_jobs_for_candidate(
                    selected_candidate['id'], 
                    top_n=top_n,
                    min_probability=min_probability
                )
                
                if not matches:
                    st.warning("⚠️ Nenhuma vaga encontrada com os critérios especificados.")
                    st.info("💡 Tente reduzir a probabilidade mínima ou verificar se há vagas compatíveis com a senioridade do candidato.")
                    return
                
                # Mostra resultados
                # Encontradas vagas compatíveis
                
                # Prepara dados para visualização
                matches_df = pd.DataFrame([
                    {
                        'vaga_id': m['vaga_id'],
                        'match_probability': m['match_probability'],
                        'vaga_nome': m['vaga_info'].get('titulo', f"Vaga {m['vaga_id']}"),
                        'area': m['vaga_info'].get('area', 'N/A'),
                        'senioridade': m['vaga_info'].get('senioridade', 'N/A'),
                        'salario': m['vaga_info'].get('salario', 0),
                        'empresa': m['vaga_info'].get('empresa', 'N/A')
                    }
                    for m in matches[:10]  # Top 10 para o gráfico
                ])
                
                # Gráfico de probabilidades
                fig = px.bar(
                    matches_df,
                    x='match_probability',
                    y='vaga_nome',
                    orientation='h',
                    title='Top 10 Vagas por Probabilidade de Match',
                    labels={'match_probability': 'Probabilidade', 'vaga_nome': 'Vaga'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela detalhada
                st.subheader("📊 Resultados Detalhados")
                
                # Prepara dados para tabela
                all_matches_df = pd.DataFrame([
                    {
                        'Vaga': m['vaga_info'].get('titulo', f"Vaga {m['vaga_id']}"),
                        'Probabilidade': f"{m['match_probability']:.1%}",
                        'Área': m['vaga_info'].get('area', 'N/A'),
                        'Senioridade': m['vaga_info'].get('senioridade', 'N/A'),
                        'Salário': f"R$ {m['vaga_info'].get('salario', 0):,.2f}",
                        'Empresa': m['vaga_info'].get('empresa', 'N/A')
                    }
                    for m in matches
                ])
                
                st.dataframe(all_matches_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao buscar vagas: {str(e)}")

def show_job_to_candidates_matching():
    """Mostra interface para encontrar candidatos para uma vaga"""
    st.subheader("💼 Encontrar Candidatos para Vaga")
    
    # Verifica se modelo está carregado
    if not (hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded):
        st.warning("⚠️ Carregue um modelo primeiro na aba 'Configuração do Modelo'.")
        return
    
    service = st.session_state.job_matching_service
    
    # Carrega dados de vagas
    try:
        jobs_df = service.get_available_jobs()
        if jobs_df.empty:
            st.warning("⚠️ Nenhuma vaga não preenchida encontrada.")
            return
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar vagas: {str(e)}")
        return
    
    # Seleção da vaga
    job_options = []
    for idx, row in jobs_df.iterrows():
        vaga_id = row.get('vaga_id', str(idx))
        titulo = row.get('titulo', f'Vaga {vaga_id}')
        area = row.get('area', 'N/A')
        senioridade = row.get('senioridade', 'N/A')
        salario = row.get('salario', 0)
        job_options.append({
            'id': vaga_id,
            'display': f"{titulo} - {area} - {senioridade} - R$ {salario:,.2f}",
            'info': row
        })
    
    selected_job_idx = st.selectbox(
        "Selecione a vaga:",
        range(len(job_options)),
        format_func=lambda x: job_options[x]['display'],
        key="selected_job"
    )
    
    selected_job = job_options[selected_job_idx]
    
    # Mostra informações da vaga
    st.info(f"📋 **Vaga Selecionada:** {selected_job['display']}")
    
    # Parâmetros de busca
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Número de candidatos a retornar:", 5, 50, 15, key="job_to_candidates_top_n")
    
    with col2:
        min_probability = st.slider("Probabilidade mínima:", 0.0, 1.0, 0.1, 0.01, key="job_to_candidates_min_prob")
    
    with col3:
        st.info("🤖 Sistema usa modelo treinado para predições reais")
    
    # Botão para buscar candidatos
    if st.button("🔍 Buscar Melhores Candidatos", type="primary", key="search_candidates_btn"):
        with st.spinner("Analisando matches com modelo de Deep Learning..."):
            try:
                matches = service.find_best_candidates_for_job(
                    selected_job['id'], 
                    top_n=top_n,
                    min_probability=min_probability
                )
                
                if not matches:
                    st.warning("⚠️ Nenhum candidato encontrado com os critérios especificados.")
                    st.info("💡 Tente reduzir a probabilidade mínima ou verificar se há candidatos compatíveis.")
                    return
                
                # Mostra resultados
                # Encontrados candidatos compatíveis
                
                # Prepara dados para visualização
                matches_df = pd.DataFrame([
                    {
                        'candidato_id': m['candidate_id'],  # Corrigido: era candidato_id mas deveria ser candidate_id
                        'match_probability': m['match_probability'],
                        'candidato_nome': m['candidate_info'].get('nome', f"Candidato {m['candidate_id']}"),
                        'area': m['candidate_info'].get('area', 'N/A'),
                        'senioridade': m['candidate_info'].get('senioridade', 'N/A'),
                        'salario_desejado': m['candidate_info'].get('salario_desejado', 0)
                    }
                    for m in matches[:10]  # Top 10 para o gráfico
                ])
                
                # Gráfico de probabilidades
                fig = px.bar(
                    matches_df,
                    x='match_probability',
                    y='candidato_nome',
                    orientation='h',
                    title='Top 10 Candidatos por Probabilidade de Match',
                    labels={'match_probability': 'Probabilidade', 'candidato_nome': 'Candidato'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabela detalhada
                st.subheader("📊 Resultados Detalhados")
                
                # Prepara dados para tabela
                all_matches_df = pd.DataFrame([
                    {
                        'Candidato': m['candidate_info'].get('nome', f"Candidato {m['candidate_id']}"),
                        'Probabilidade': f"{m['match_probability']:.1%}",
                        'Área': m['candidate_info'].get('area', 'N/A'),
                        'Senioridade': m['candidate_info'].get('senioridade', 'N/A'),
                        'Salário Desejado': f"R$ {m['candidate_info'].get('salario_desejado', 0):,.2f}"
                    }
                    for m in matches
                ])
                
                st.dataframe(all_matches_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Erro ao buscar candidatos: {str(e)}")

if __name__ == "__main__":
    create_job_matching_interface()