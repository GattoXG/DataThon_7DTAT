import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def create_advanced_analysis(df_vagas, df_prospects):
    """Cria análises avançadas dos dados"""
    
    st.header("🔬 Análises Avançadas")
    st.markdown("*Análises específicas de correlações, performance temporal e insights automáticos*")
    
    # Tabs para organizar melhor
    tab1, tab2, tab3 = st.tabs(["📅 Análise Temporal", "🔗 Correlações", "💡 Insights Automáticos"])
    
    with tab1:
        st.subheader("📅 Análise Temporal")
        
        # Converter datas para análise temporal
        df_prospects_temp = df_prospects.copy()
        df_prospects_temp['data_candidatura_parsed'] = pd.to_datetime(
            df_prospects_temp['data_candidatura'], 
            format='%d-%m-%Y', 
            errors='coerce'
        )
        
        # Filtrar apenas dados com datas válidas
        df_with_dates = df_prospects_temp.dropna(subset=['data_candidatura_parsed'])
        
        if not df_with_dates.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Análise de candidaturas por mês
                df_with_dates['mes_ano'] = df_with_dates['data_candidatura_parsed'].dt.to_period('M').astype(str)
                candidaturas_mes = df_with_dates.groupby('mes_ano').size().reset_index(name='count')
                
                fig_temporal = px.line(
                    candidaturas_mes, 
                    x='mes_ano', 
                    y='count',
                    title='📈 Evolução de Candidaturas por Mês',
                    markers=True
                )
                fig_temporal.update_xaxes(tickangle=45)
                st.plotly_chart(fig_temporal, use_container_width=True)
            
            with col2:
                # Taxa de conversão por mês
                conversao_mes = df_with_dates.groupby('mes_ano').agg({
                    'situacao': ['count', lambda x: sum('Contratado' in str(s) for s in x)]
                }).round(2)
                conversao_mes.columns = ['Total', 'Contratados']
                conversao_mes['Taxa_Conversao'] = (conversao_mes['Contratados'] / conversao_mes['Total'] * 100).round(2)
                conversao_mes = conversao_mes.reset_index()
                
                fig_conversao = px.bar(
                    conversao_mes,
                    x='mes_ano',
                    y='Taxa_Conversao',
                    title='📊 Taxa de Conversão por Mês (%)',
                    color='Taxa_Conversao',
                    color_continuous_scale='viridis'
                )
                fig_conversao.update_xaxes(tickangle=45)
                st.plotly_chart(fig_conversao, use_container_width=True)
            
            # Análise por dia da semana
            df_with_dates['dia_semana'] = df_with_dates['data_candidatura_parsed'].dt.day_name()
            dia_semana_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            candidaturas_dia = df_with_dates.groupby('dia_semana').size().reindex(dia_semana_order).reset_index(name='count')
            
            fig_dia_semana = px.bar(
                candidaturas_dia,
                x='dia_semana',
                y='count',
                title='📅 Candidaturas por Dia da Semana',
                color='count',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_dia_semana, use_container_width=True)
            
        else:
            st.info("📅 Dados de data não disponíveis para análise temporal")
    
    with tab2:
        st.subheader("🔗 Análise de Correlações")
        
        # Merge dos dados para análise
        vagas_prospects = df_prospects.merge(
            df_vagas, 
            left_on='codigo_vaga', 
            right_on='codigo_vaga', 
            how='left'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Análise por tipo de vaga SAP
            if 'vaga_sap' in vagas_prospects.columns:
                sap_analysis = vagas_prospects.groupby('vaga_sap').agg({
                    'situacao': ['count', lambda x: sum('Contratado' in str(s) for s in x)]
                })
                sap_analysis.columns = ['Total_Prospects', 'Contratados']
                sap_analysis['Taxa_Conversao'] = (sap_analysis['Contratados'] / sap_analysis['Total_Prospects'] * 100).round(2)
                
                fig_sap = px.bar(
                    x=sap_analysis.index,
                    y=sap_analysis['Taxa_Conversao'],
                    title='💼 Taxa de Conversão: SAP vs Não SAP',
                    color=sap_analysis['Taxa_Conversao'],
                    color_continuous_scale='oranges'
                )
                st.plotly_chart(fig_sap, use_container_width=True)
                
                st.dataframe(sap_analysis.style.format({'Taxa_Conversao': '{:.1f}%'}))
        
        with col2:
            # Análise por nível profissional
            if 'nivel_profissional' in vagas_prospects.columns:
                nivel_analysis = vagas_prospects.groupby('nivel_profissional').agg({
                    'situacao': ['count', lambda x: sum('Contratado' in str(s) for s in x)]
                })
                nivel_analysis.columns = ['Total_Prospects', 'Contratados']
                nivel_analysis['Taxa_Conversao'] = (nivel_analysis['Contratados'] / nivel_analysis['Total_Prospects'] * 100).round(2)
                nivel_analysis = nivel_analysis.sort_values('Taxa_Conversao', ascending=False)
                
                fig_nivel = px.bar(
                    x=nivel_analysis['Taxa_Conversao'],
                    y=nivel_analysis.index,
                    orientation='h',
                    title='📊 Taxa de Conversão por Nível Profissional',
                    color=nivel_analysis['Taxa_Conversao'],
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_nivel, use_container_width=True)
        
        # Análise de performance de recrutadores
        st.subheader("👥 Performance de Recrutadores")
        
        recruiter_performance = df_prospects.groupby('recrutador').agg({
            'situacao': [
                'count', 
                lambda x: sum('Contratado' in str(s) for s in x),
                lambda x: sum('Encaminhado' in str(s) for s in x)
            ]
        })
        recruiter_performance.columns = ['Total_Prospects', 'Contratados', 'Encaminhados']
        recruiter_performance['Taxa_Conversao'] = (recruiter_performance['Contratados'] / recruiter_performance['Total_Prospects'] * 100).round(2)
        recruiter_performance['Taxa_Encaminhamento'] = (recruiter_performance['Encaminhados'] / recruiter_performance['Total_Prospects'] * 100).round(2)
        
        # Filtrar recrutadores com volume mínimo
        recruiter_performance = recruiter_performance[recruiter_performance['Total_Prospects'] >= 5]
        recruiter_performance = recruiter_performance.sort_values('Taxa_Conversao', ascending=False)
        
        if not recruiter_performance.empty:
            # Scatter plot: Volume vs Taxa de Conversão
            fig_scatter = px.scatter(
                x=recruiter_performance['Total_Prospects'],
                y=recruiter_performance['Taxa_Conversao'],
                hover_name=recruiter_performance.index,
                title='📊 Volume vs Taxa de Conversão por Recrutador',
                labels={'x': 'Total de Prospects', 'y': 'Taxa de Conversão (%)'},
                color=recruiter_performance['Taxa_Conversao'],
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Top performers
            st.subheader("🏆 Top Performers")
            top_performers = recruiter_performance.head(10)
            st.dataframe(top_performers.style.format({
                'Taxa_Conversao': '{:.1f}%',
                'Taxa_Encaminhamento': '{:.1f}%'
            }))
        else:
            st.info("👥 Dados insuficientes para análise de recrutadores (mínimo 5 prospects)")
    
    with tab3:
        st.subheader("💡 Insights Automáticos")
        
        insights = []
        
        # Insight sobre melhor taxa de conversão
        if not recruiter_performance.empty:
            best_recruiter = recruiter_performance.index[0]
            best_rate = recruiter_performance.iloc[0]['Taxa_Conversao']
            insights.append(f"🎯 **Melhor recrutador:** {best_recruiter} com {best_rate:.1f}% de taxa de conversão")
        
        # Insight sobre volume
        total_prospects = len(df_prospects)
        total_contratados = len(df_prospects[df_prospects['situacao'].str.contains('Contratado', na=False)])
        if total_prospects > 0:
            taxa_geral = (total_contratados / total_prospects) * 100
            insights.append(f"📊 **Taxa geral de conversão:** {taxa_geral:.1f}%")
        
        # Insight sobre SAP
        if 'vaga_sap' in vagas_prospects.columns and 'sap_analysis' in locals():
            if len(sap_analysis) >= 2:
                sap_sim = sap_analysis.loc['Sim', 'Taxa_Conversao'] if 'Sim' in sap_analysis.index else 0
                sap_nao = sap_analysis.loc['Não', 'Taxa_Conversao'] if 'Não' in sap_analysis.index else 0
                
                if sap_sim > sap_nao:
                    insights.append(f"💼 **Vagas SAP** têm melhor conversão ({sap_sim:.1f}% vs {sap_nao:.1f}%)")
                elif sap_nao > sap_sim:
                    insights.append(f"💼 **Vagas não-SAP** têm melhor conversão ({sap_nao:.1f}% vs {sap_sim:.1f}%)")
        
        # Insight sobre melhor nível profissional
        if 'nivel_analysis' in locals() and not nivel_analysis.empty:
            melhor_nivel = nivel_analysis.index[0]
            melhor_taxa = nivel_analysis.iloc[0]['Taxa_Conversao']
            insights.append(f"🎓 **Melhor nível profissional:** {melhor_nivel} com {melhor_taxa:.1f}% de conversão")
        
        # Insight temporal
        if 'candidaturas_mes' in locals() and not candidaturas_mes.empty:
            mes_pico = candidaturas_mes.loc[candidaturas_mes['count'].idxmax(), 'mes_ano']
            volume_pico = candidaturas_mes['count'].max()
            insights.append(f"📅 **Mês com maior volume:** {mes_pico} com {volume_pico} candidaturas")
        
        # Exibe insights
        if insights:
            for insight in insights:
                st.write(insight)
                st.markdown("---")
        else:
            pass  # Não foi possível gerar insights automáticos
        
        # Recomendações
        st.subheader("🚀 Recomendações")
        
        recomendacoes = [
            "📈 **Foque nos recrutadores de alto desempenho** - Analise suas estratégias e replique",
            "🎯 **Priorize vagas com melhor taxa de conversão** - Concentre esforços onde há mais sucesso",
            "📅 **Monitore sazonalidade** - Ajuste estratégias baseado nos padrões temporais",
            "💼 **Analise diferenças entre tipos de vaga** - SAP vs não-SAP podem ter abordagens diferentes",
            "👥 **Capacite recrutadores com menor performance** - Compartilhe melhores práticas"
        ]
        
        for rec in recomendacoes:
            st.write(rec)
    
    st.markdown("---")

if __name__ == "__main__":
    st.title("Análises Avançadas")
    st.write("Este módulo precisa ser executado a partir da aplicação principal.")
    st.markdown("---")

if __name__ == "__main__":
    st.title("Análises Avançadas")
    st.write("Este módulo precisa ser executado a partir da aplicação principal.")