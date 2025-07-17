#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Página para treinamento e uso do modelo de Deep Learning
Interface para job matching com redes neurais
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_deep_learning_interface(df_vagas_dl, df_prospects_dl, df_candidates_dl):
    """
    Cria interface para treinamento e uso do modelo de Deep Learning
    
    Args:
        df_vagas_dl: DataFrame de vagas processado para Deep Learning
        df_prospects_dl: DataFrame de prospects processado para Deep Learning
        df_candidates_dl: DataFrame de candidatos processado para Deep Learning
    """
    
    st.header("🧠 Deep Learning - Job Matching")
    
    # Verifica se os DataFrames não estão vazios
    if df_vagas_dl.empty or df_prospects_dl.empty or df_candidates_dl.empty:
        st.error("Dados não foram processados corretamente. Verifique o feature engineering.")
        return
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs([
        "📊 Análise dos Dados", 
        "🎯 Treinamento do Modelo", 
        " Análise do Modelo"
    ])
    
    with tab1:
        show_data_analysis(df_vagas_dl, df_prospects_dl, df_candidates_dl)
    
    with tab2:
        show_model_training(df_vagas_dl, df_prospects_dl, df_candidates_dl)
    
    with tab3:
        show_model_analysis()

def show_data_analysis(df_vagas_dl, df_prospects_dl, df_candidates_dl):
    """Mostra análise dos dados processados"""
    st.subheader("📊 Análise dos Dados Processados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Vagas Processadas", len(df_vagas_dl))
        st.metric("Features de Vagas", len(df_vagas_dl.columns) - 1)  # -1 para excluir ID
    
    with col2:
        st.metric("Prospects Processados", len(df_prospects_dl))
        st.metric("Features de Prospects", len(df_prospects_dl.columns) - 2)  # -2 para excluir IDs
    
    with col3:
        st.metric("Candidatos Processados", len(df_candidates_dl))
        st.metric("Features de Candidatos", len(df_candidates_dl.columns) - 1)  # -1 para excluir ID
    
    # Criação do dataset de treinamento
    st.subheader("📋 Dataset de Treinamento")
    
    try:
        from services.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        
        # Cria dataset de treinamento
        with st.spinner("Criando dataset de treinamento..."):
            training_data = fe.create_training_dataset(
                df_vagas_dl, df_prospects_dl, df_candidates_dl
            )
        
        if not training_data.empty:
            # Informações do dataset
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Registros", len(training_data))
                st.metric("Total de Features", len(training_data.columns) - 1)  # -1 para target
            
            with col2:
                if 'target' in training_data.columns:
                    target_dist = training_data['target'].value_counts()
                    contratacoes = target_dist.get(1, 0)
                    nao_contratacoes = target_dist.get(0, 0)
                    
                    st.metric("Contratações (1)", contratacoes)
                    st.metric("Não Contratações (0)", nao_contratacoes)
                    
                    # Mostra percentual de contratações
                    total_registros = len(training_data)
                    percentual = (contratacoes / total_registros) * 100 if total_registros > 0 else 0
                    st.metric("Taxa de Contratação", f"{percentual:.2f}%")
            
            # Gráfico de distribuição do target
            if 'target' in training_data.columns:
                target_dist = training_data['target'].value_counts()
                
                # Só cria gráfico se há pelo menos uma categoria
                if len(target_dist) > 0:
                    # Garante que temos labels para todas as categorias
                    labels = []
                    values = []
                    
                    for idx, count in target_dist.items():
                        if idx == 0:
                            labels.append('Não Contratado')
                        elif idx == 1:
                            labels.append('Contratado')
                        else:
                            labels.append(f'Categoria {idx}')
                        values.append(count)
                    
                    # Só cria gráfico se há mais de uma categoria ou se há contratações
                    if len(values) > 1 or (len(values) == 1 and target_dist.get(1, 0) > 0):
                        fig_target = px.pie(
                            values=values,
                            names=labels,
                            title="Distribuição do Target"
                        )
                        st.plotly_chart(fig_target, use_container_width=True)
                    else:
                        pass  # Todos os registros são 'Não Contratado'
                
                # Mostra distribuição das situações
                st.write("**Distribuição do Target:**")
                target_counts = training_data['target'].value_counts()
                for idx, count in target_counts.items():
                    label = "Contratado" if idx == 1 else "Não Contratado"
                    st.write(f"- {label}: {count} registros ({count/len(training_data)*100:.1f}%)")
            
            # Mostra amostra dos dados
            st.subheader("🔍 Amostra dos Dados")
            st.dataframe(training_data.head())
            
            # Salva dataset para uso posterior
            st.session_state['training_data'] = training_data
            
        else:
            st.error("Não foi possível criar o dataset de treinamento.")
            
    except Exception as e:
        st.error(f"Erro ao criar dataset: {str(e)}")

def show_model_training(df_vagas_dl, df_prospects_dl, df_candidates_dl):
    """Interface para treinamento do modelo"""
    st.subheader("🎯 Treinamento do Modelo")
    
    # Parâmetros do modelo
    st.subheader("⚙️ Configurações do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Arquitetura da Rede:**")
        num_layers = st.slider("Número de Camadas Ocultas", 2, 5, 3)
        
        hidden_sizes = []
        for i in range(num_layers):
            size = st.slider(f"Camada {i+1}", 32, 512, 256 // (i+1))
            hidden_sizes.append(size)
        
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3)
    
    with col2:
        st.write("**Parâmetros de Treinamento:**")
        learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
        epochs = st.slider("Épocas", 50, 300, 150)  # Mínimo maior para dados desbalanceados
        patience = st.slider("Early Stopping Patience", 15, 40, 30)  # Mais tolerante
    
    # Botão para iniciar treinamento
    if st.button("🚀 Iniciar Treinamento", type="primary"):
        
        if 'training_data' not in st.session_state:
            st.error("Primeiro vá para a aba 'Análise dos Dados' para criar o dataset.")
            return
        
        training_data = st.session_state['training_data']
        
        # Verifica se há contratações suficientes e mostra estatísticas
        if 'target' in training_data.columns:
            target_dist = training_data['target'].value_counts()
            contratacoes = target_dist.get(1, 0)
            nao_contratacoes = target_dist.get(0, 0)
            total = contratacoes + nao_contratacoes
            
            # Mostra estatísticas de desbalanceamento
            st.subheader("📊 Análise do Dataset")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Contratações", contratacoes)
            with col2:
                st.metric("Não Contratações", nao_contratacoes)
            with col3:
                ratio = nao_contratacoes / contratacoes if contratacoes > 0 else 0
                st.metric("Proporção", f"1:{ratio:.1f}")
            
            # Alerta sobre desbalanceamento
            if ratio > 10:
                pass  # Dataset muito desbalanceado
            
            if contratacoes == 0:
                st.error("❌ Não há contratações no dataset. O modelo não pode ser treinado.")
                return
            elif contratacoes < 100:
                st.warning(f"⚠️ Poucas contratações encontradas ({contratacoes}). Recomenda-se pelo menos 100 para um modelo robusto.")
                if not st.checkbox("Treinar mesmo assim"):
                    return
        
        try:
            # Importa o modelo
            from services.deep_learning_model import DeepLearningModel
            
            # Cria modelo
            model = DeepLearningModel(
                hidden_sizes=hidden_sizes,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate
            )
            
            # Barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Prepara dados
            status_text.text("Preparando dados...")
            progress_bar.progress(20)
            
            train_loader, test_loader, X_train, X_test = model.prepare_data(training_data)
            
            # Cria modelo
            status_text.text("Criando modelo...")
            progress_bar.progress(40)
            
            model.create_model(X_train.shape[1])
            
            # Treina modelo
            status_text.text("Treinando modelo...")
            progress_bar.progress(60)
            
            training_results = model.train_model(
                train_loader, test_loader, 
                epochs=epochs, patience=patience
            )
            
            # Avalia modelo
            status_text.text("Avaliando modelo...")
            progress_bar.progress(80)
            
            evaluation_results = model.evaluate_model(test_loader)
            
            # Salva modelo
            status_text.text("Salvando modelo...")
            progress_bar.progress(90)
            
            model_path = f"models/job_matching_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs("models", exist_ok=True)
            model.save_model(model_path, evaluation_results, training_results)
            
            progress_bar.progress(100)
            status_text.text("Treinamento concluído!")
            
            # Salva modelo e resultados na sessão
            st.session_state['trained_model'] = model
            st.session_state['training_results'] = training_results
            st.session_state['evaluation_results'] = evaluation_results
            
            # Mostra resultados
            # Modelo treinado com sucesso
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Treinamento")
                st.metric("Melhor Acurácia de Validação", f"{training_results['best_val_accuracy']:.4f}")
                st.metric("Épocas Treinadas", training_results['total_epochs'])
            
            with col2:
                st.subheader("📈 Métricas Gerais")
                st.metric("Acurácia", f"{evaluation_results['accuracy']:.4f}")
                st.metric("ROC AUC", f"{evaluation_results['roc_auc']:.4f}")
                
            with col3:
                st.subheader("🎯 Métricas de Classe")
                st.metric("F1-Score", f"{evaluation_results['f1_score']:.4f}")
                st.metric("Precisão", f"{evaluation_results.get('precision', 0):.4f}")
                st.metric("Recall", f"{evaluation_results.get('recall', 0):.4f}")
            
            # Mostra contagem de predições por classe
            if 'predictions' in evaluation_results:
                pred_counts = np.bincount(evaluation_results['predictions'])
                st.subheader("📋 Distribuição das Predições")
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    st.metric("Predições 'Não Contratado'", pred_counts[0] if len(pred_counts) > 0 else 0)
                with pred_col2:
                    st.metric("Predições 'Contratado'", pred_counts[1] if len(pred_counts) > 1 else 0)
            
            # Gráfico de histórico de treinamento
            show_training_history(training_results['training_history'])
            
        except Exception as e:
            st.error(f"Erro durante o treinamento: {str(e)}")
            st.exception(e)

def show_training_history(history):
    """Mostra histórico de treinamento"""
    st.subheader("📈 Histórico de Treinamento")
    
    history_df = pd.DataFrame(history)
    
    # Gráfico de loss
    fig_loss = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy')
    )
    
    # Loss
    fig_loss.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['train_loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig_loss.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['val_loss'],
            mode='lines',
            name='Val Loss',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Accuracy
    fig_loss.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['train_acc'],
            mode='lines',
            name='Train Acc',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    fig_loss.add_trace(
        go.Scatter(
            x=history_df['epoch'],
            y=history_df['val_acc'],
            mode='lines',
            name='Val Acc',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    fig_loss.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_loss, use_container_width=True)

def show_model_analysis():
    """Mostra análise do modelo treinado"""
    st.subheader("📈 Análise do Modelo")
    
    # Função para listar modelos salvos
    def get_saved_models():
        """Lista todos os modelos salvos na pasta models"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            return []
        
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and file.startswith('job_matching_model_'):
                # Extrai timestamp do nome do arquivo
                timestamp_str = file.replace('job_matching_model_', '').replace('.pkl', '')
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    model_files.append({
                        'filename': file,
                        'filepath': os.path.join(models_dir, file),
                        'timestamp': timestamp,
                        'display_name': f"Modelo {timestamp.strftime('%d/%m/%Y %H:%M:%S')}"
                    })
                except ValueError:
                    continue
        
        # Ordena por timestamp (mais recente primeiro)
        model_files.sort(key=lambda x: x['timestamp'], reverse=True)
        return model_files
    
    # Seleção de modelo
    st.subheader("🎯 Seleção de Modelo")
    
    saved_models = get_saved_models()
    
    if not saved_models:
        return
    
    # Opções de seleção
    model_options = ["Modelo atual da sessão"] + [model['display_name'] for model in saved_models]
    
    selected_option = st.selectbox(
        "Selecione o modelo para análise:",
        options=model_options,
        index=0
    )
    
    # Carrega o modelo selecionado
    if selected_option == "Modelo atual da sessão":
        if 'trained_model' not in st.session_state:
            return
        
        model = st.session_state['trained_model']
        training_results = st.session_state.get('training_results', {})
        evaluation_results = st.session_state.get('evaluation_results', {})
        
    else:
        # Carrega modelo salvo
        selected_model = next(m for m in saved_models if m['display_name'] == selected_option)
        
        try:
            with st.spinner(f"Carregando modelo: {selected_option}"):
                from services.deep_learning_model import DeepLearningModel
                model = DeepLearningModel()
                saved_data = model.load_model(selected_model['filepath'])
                
                # Carrega resultados salvos se disponíveis
                training_results = saved_data.get('training_results', {})
                evaluation_results = saved_data.get('evaluation_results', {})
                save_timestamp = saved_data.get('save_timestamp', None)
                
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {str(e)}")
            return
    
    # Resumo do modelo
    st.subheader("📋 Resumo do Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Arquitetura", f"{len(model.hidden_sizes)} camadas ocultas")
        st.metric("Total de Features", len(model.feature_names) if model.feature_names else "N/A")
    
    with col2:
        if evaluation_results:
            st.metric("Acurácia Final", f"{evaluation_results.get('accuracy', 0):.4f}")
            st.metric("F1-Score", f"{evaluation_results.get('f1_score', 0):.4f}")
        else:
            st.metric("Acurácia Final", "N/A - Modelo carregado")
            st.metric("F1-Score", "N/A - Modelo carregado")
    
    with col3:
        if evaluation_results:
            st.metric("ROC AUC", f"{evaluation_results.get('roc_auc', 0):.4f}")
        else:
            st.metric("ROC AUC", "N/A - Modelo carregado")
            
        if training_results:
            st.metric("Épocas Treinadas", training_results.get('total_epochs', 0))
        else:
            st.metric("Épocas Treinadas", "N/A - Modelo carregado")
    
    # Informações adicionais do modelo
    st.subheader("🔧 Configurações do Modelo")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.write(f"**Taxa de Dropout:** {model.dropout_rate}")
        st.write(f"**Taxa de Aprendizado:** {model.learning_rate}")
        st.write(f"**Camadas Ocultas:** {model.hidden_sizes}")
    
    with config_col2:
        st.write(f"**Modelo Treinado:** {'Sim' if model.is_trained else 'Não'}")
        if hasattr(model, 'training_history') and model.training_history:
            st.write(f"**Histórico de Treinamento:** {len(model.training_history)} épocas")
        else:
            st.write("**Histórico de Treinamento:** Não disponível")
    
    # Gráficos de avaliação se há dados disponíveis
    if evaluation_results:
        # Matriz de confusão
        show_confusion_matrix(evaluation_results)
        
        # Curva ROC
        show_roc_curve(evaluation_results)
        
        # Métricas adicionais
        st.subheader("� Métricas Detalhadas")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Precisão", f"{evaluation_results.get('precision', 0):.4f}")
            st.metric("Recall", f"{evaluation_results.get('recall', 0):.4f}")
        
        with metrics_col2:
            st.metric("Acurácia", f"{evaluation_results.get('accuracy', 0):.4f}")
            st.metric("F1-Score", f"{evaluation_results.get('f1_score', 0):.4f}")
            
    else:
        pass  # Este modelo não possui resultados de avaliação salvos
    
    # Histórico de treinamento se disponível
    if hasattr(model, 'training_history') and model.training_history:
        st.subheader("📈 Histórico de Treinamento")
        
        # Converte histórico para DataFrame
        history_df = pd.DataFrame(model.training_history)
        
        if not history_df.empty:
            # Gráfico de loss
            fig_loss = px.line(
                history_df, 
                x=history_df.index, 
                y=['train_loss', 'val_loss'],
                title="Evolução da Loss durante o Treinamento",
                labels={'index': 'Época', 'value': 'Loss'}
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Gráfico de acurácia se disponível
            if 'train_accuracy' in history_df.columns:
                fig_acc = px.line(
                    history_df, 
                    x=history_df.index, 
                    y=['train_accuracy', 'val_accuracy'],
                    title="Evolução da Acurácia durante o Treinamento",
                    labels={'index': 'Época', 'value': 'Acurácia'}
                )
                st.plotly_chart(fig_acc, use_container_width=True)

def show_confusion_matrix(evaluation_results):
    """Mostra matriz de confusão"""
    from sklearn.metrics import confusion_matrix
    
    st.subheader("🎯 Matriz de Confusão")
    
    y_true = evaluation_results['targets']
    y_pred = evaluation_results['predictions']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Matriz de Confusão",
        labels=dict(x="Predições", y="Valores Reais", color="Contagem")
    )
    
    fig_cm.update_xaxes(tickvals=[0, 1], ticktext=["Não Contratado", "Contratado"])
    fig_cm.update_yaxes(tickvals=[0, 1], ticktext=["Não Contratado", "Contratado"])
    
    st.plotly_chart(fig_cm, use_container_width=True)

def show_roc_curve(evaluation_results):
    """Mostra curva ROC"""
    from sklearn.metrics import roc_curve
    
    st.subheader("📈 Curva ROC")
    
    y_true = evaluation_results['targets']
    y_probs = [prob[1] for prob in evaluation_results['probabilities']]
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {evaluation_results["roc_auc"]:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='red', width=1, dash='dash')
    ))
    
    fig_roc.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        showlegend=True
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
