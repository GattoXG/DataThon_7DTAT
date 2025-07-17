import json
import pandas as pd
import streamlit as st
from .salary_converter import SalaryConverter
from .feature_engineering import FeatureEngineering

# Inicializa o conversor de salários
salary_converter = SalaryConverter()

# Inicializa o processador de feature engineering
feature_engineer = FeatureEngineering()

@st.cache_data
def load_data():
    """Carrega todos os dados JSON"""
    try:
        with open('data/vagas.json', 'r', encoding='utf-8') as f:
            vagas = json.load(f)
        
        with open('data/prospects.json', 'r', encoding='utf-8') as f:
            prospects = json.load(f)

        with open('data/applicants.json', 'r', encoding='utf-8') as f:
            applicants = json.load(f)
        
        return vagas, prospects, applicants
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return {}, {}, {}

def process_vagas_data(vagas):
    """Processa dados das vagas para análise"""
    vagas_list = []
    
    for codigo, vaga in vagas.items():
        try:
            vaga_info = {
                'codigo_vaga': codigo,
                'titulo': vaga.get('informacoes_basicas', {}).get('titulo_vaga', ''),
                'cliente': vaga.get('informacoes_basicas', {}).get('cliente', ''),
                'vaga_sap': vaga.get('informacoes_basicas', {}).get('vaga_sap', ''),
                'tipo_contratacao': vaga.get('informacoes_basicas', {}).get('tipo_contratacao', ''),
                'analista_responsavel': vaga.get('informacoes_basicas', {}).get('analista_responsavel', ''),
                'estado': vaga.get('perfil_vaga', {}).get('estado', ''),
                'cidade': vaga.get('perfil_vaga', {}).get('cidade', ''),
                'nivel_profissional': vaga.get('perfil_vaga', {}).get('nivel profissional', ''),
                'nivel_academico': vaga.get('perfil_vaga', {}).get('nivel_academico', ''),
                'nivel_ingles': vaga.get('perfil_vaga', {}).get('nivel_ingles', ''),
                'nivel_espanhol': vaga.get('perfil_vaga', {}).get('nivel_espanhol', ''),
                'areas_atuacao': vaga.get('perfil_vaga', {}).get('areas_atuacao', ''),
                'data_requisicao': vaga.get('informacoes_basicas', {}).get('data_requicisao', ''),
                'valor_venda': vaga.get('beneficios', {}).get('valor_venda', '')
            }
            vagas_list.append(vaga_info)
        except Exception:
            continue
    
    df_vagas = pd.DataFrame(vagas_list)
    
    # Processa valores de venda usando o SalaryConverter
    if not df_vagas.empty and 'valor_venda' in df_vagas.columns:
        # Processa em um DataFrame temporário para evitar problemas de colunas
        temp_df = salary_converter.process_salary_dataframe(df_vagas.copy(), 'valor_venda')
        
        # Lógica mais tolerante: usa valores válidos, mas tenta recuperar alguns inválidos
        def get_best_salary_value(row):
            # Se é válido, usa o valor processado
            if row['salary_is_valid']:
                return row['salary_monthly_value_brl']
            
            # Se inválido, tenta recuperar casos específicos
            original = str(row['salary_original']).strip().lower()
            
            # Casos recuperáveis:
            # 1. Apenas "-" -> assumir como não informado (0.0)
            if original == '-':
                return 0.0
            
            # 2. Textos que indicam "a combinar" -> assumir como não informado (0.0)
            if any(phrase in original for phrase in ['combinar', 'negociar', 'n/a', 'negociavel']):
                return 0.0
            
            # 3. Se tem valor numérico mas foi marcado como inválido por ser muito alto/baixo
            if row['salary_raw_value'] is not None:
                raw_value = row['salary_raw_value']
                
                # Se o valor bruto parece razoável (entre 500 e 50000)
                if 500 <= raw_value <= 50000:
                    return raw_value
                
                # Se o valor é muito pequeno, pode ser valor por hora
                if raw_value < 500:
                    return raw_value * 220  # Converte para mensal (220h/mês)
            
            # Se não conseguiu recuperar, retorna None
            return None
        
        # Aplica a lógica melhorada
        df_vagas['valor_venda_num'] = temp_df.apply(get_best_salary_value, axis=1)
    
    return df_vagas

def process_prospects_data(prospects):
    """Processa dados dos prospects para análise"""
    prospects_list = []
    
    for codigo_vaga, vaga_data in prospects.items():
        for prospect in vaga_data.get('prospects', []):
            try:
                prospect_info = {
                    'codigo_vaga': codigo_vaga,
                    'titulo_vaga': vaga_data.get('titulo', ''),
                    'nome_candidato': prospect.get('nome', ''),
                    'codigo_candidato': prospect.get('codigo', ''),
                    'situacao': prospect.get('situacao_candidado', ''),
                    'data_candidatura': prospect.get('data_candidatura', ''),
                    'ultima_atualizacao': prospect.get('ultima_atualizacao', ''),
                    'comentario': prospect.get('comentario', ''),
                    'recrutador': prospect.get('recrutador', '')
                }
                prospects_list.append(prospect_info)
            except Exception:
                continue
    
    return pd.DataFrame(prospects_list)

def process_applicants_data(applicants):
    """Processa dados dos candidatos para análise"""
    applicants_list = []
    
    for codigo, candidato in applicants.items():
        try:
            applicant_info = {
                'codigo': codigo,
                'nome': candidato.get('infos_basicas', {}).get('nome', '') or candidato.get('informacoes_pessoais', {}).get('nome', ''),
                'email': candidato.get('infos_basicas', {}).get('email', '') or candidato.get('informacoes_pessoais', {}).get('email', ''),
                'telefone': candidato.get('infos_basicas', {}).get('telefone', '') or candidato.get('informacoes_pessoais', {}).get('telefone_celular', ''),
                'objetivo_profissional': candidato.get('infos_basicas', {}).get('objetivo_profissional', ''),
                'nivel_academico': candidato.get('formacao_e_idiomas', {}).get('nivel_academico', ''),
                'nivel_ingles': candidato.get('formacao_e_idiomas', {}).get('nivel_ingles', ''),
                'nivel_espanhol': candidato.get('formacao_e_idiomas', {}).get('nivel_espanhol', ''),
                'area_atuacao': candidato.get('informacoes_profissionais', {}).get('area_atuacao', ''),
                'remuneracao': candidato.get('infos_basicas', {}).get('remuneracao', '') or candidato.get('informacoes_profissionais', {}).get('remuneracao', ''),
                'data_criacao': candidato.get('infos_basicas', {}).get('data_criacao', ''),
                'inserido_por': candidato.get('infos_basicas', {}).get('inserido_por', ''),
                'titulo_profissional': candidato.get('informacoes_profissionais', {}).get('titulo_profissional', ''),
                'conhecimentos_tecnicos': candidato.get('informacoes_profissionais', {}).get('conhecimentos_tecnicos', ''),
                'certificacoes': candidato.get('informacoes_profissionais', {}).get('certificacoes', '')
            }
            applicants_list.append(applicant_info)
        except Exception:
            continue
    
    df_applicants = pd.DataFrame(applicants_list)
    
    # Processa salários usando o SalaryConverter
    if not df_applicants.empty and 'remuneracao' in df_applicants.columns:
        # Processa em um DataFrame temporário para evitar problemas de colunas
        temp_df = salary_converter.process_salary_dataframe(df_applicants.copy(), 'remuneracao')
        
        # Lógica mais tolerante: usa valores válidos, mas tenta recuperar alguns inválidos
        def get_best_salary_value(row):
            # Se é válido, usa o valor processado
            if row['salary_is_valid']:
                return row['salary_monthly_value_brl']
            
            # Se inválido, tenta recuperar casos específicos
            original = str(row['salary_original']).strip().lower()
            
            # Casos recuperáveis:
            # 1. Apenas "-" -> assumir como não informado (0.0)
            if original == '-':
                return 0.0
            
            # 2. Textos que indicam "a combinar" -> assumir como não informado (0.0)
            if any(phrase in original for phrase in ['combinar', 'negociar', 'n/a', 'negociavel']):
                return 0.0
            
            # 3. Se tem valor numérico mas foi marcado como inválido por ser muito alto/baixo
            if row['salary_raw_value'] is not None:
                raw_value = row['salary_raw_value']
                
                # Se o valor bruto parece razoável (entre 500 e 50000)
                if 500 <= raw_value <= 50000:
                    return raw_value
                
                # Se o valor é muito pequeno, pode ser valor por hora
                if raw_value < 500:
                    return raw_value * 220  # Converte para mensal (220h/mês)
            
            # Se não conseguiu recuperar, retorna None
            return None
        
        # Aplica a lógica melhorada
        df_applicants['remuneracao_num'] = temp_df.apply(get_best_salary_value, axis=1)
    
    return df_applicants

@st.cache_data
def process_vagas_for_deep_learning(vagas):
    """Processa dados das vagas para Deep Learning"""
    df_vagas = process_vagas_data(vagas)
    if df_vagas.empty:
        return pd.DataFrame()
    
    return feature_engineer.process_vagas_for_dl(df_vagas)

@st.cache_data
def process_prospects_for_deep_learning(prospects):
    """Processa dados dos prospects para Deep Learning"""
    df_prospects = process_prospects_data(prospects)
    if df_prospects.empty:
        return pd.DataFrame()
    
    return feature_engineer.process_prospects_for_dl(df_prospects)

@st.cache_data
def process_candidates_for_deep_learning(applicants):
    """Processa dados dos candidatos para Deep Learning"""
    df_candidates = process_applicants_data(applicants)
    if df_candidates.empty:
        return pd.DataFrame()
    
    return feature_engineer.process_candidates_for_dl(df_candidates)

def get_feature_engineering_summary():
    """Retorna resumo das features criadas"""
    return feature_engineer.get_feature_summary()

def save_feature_engineering_objects(filepath):
    """Salva objetos de pré-processamento"""
    feature_engineer.save_preprocessing_objects(filepath)

def load_feature_engineering_objects(filepath):
    """Carrega objetos de pré-processamento"""
    feature_engineer.load_preprocessing_objects(filepath)