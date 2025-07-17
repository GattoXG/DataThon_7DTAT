#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para feature engineering e normalização de dados para Deep Learning
Processa dados de vagas, prospects e candidatos para treinamento de modelos
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """Classe para processamento e normalização de dados para Deep Learning"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.imputers = {}
        self.feature_names = {}
        
        # Stop words em português
        self.portuguese_stop_words = [
            'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo',
            'as', 'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas',
            'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 'ele',
            'eles', 'em', 'entre', 'era', 'eram', 'essa', 'essas', 'esse',
            'esses', 'esta', 'está', 'estamos', 'estão', 'estar', 'estas',
            'estava', 'estavam', 'este', 'esteja', 'estejam', 'estejamos',
            'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera',
            'estiveram', 'estiverem', 'estivermos', 'estivesse', 'estivessem',
            'estivéramos', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos',
            'for', 'fora', 'foram', 'forem', 'formos', 'fosse', 'fossem',
            'fui', 'fôramos', 'fôssemos', 'haja', 'hajam', 'hajamos', 'havemos',
            'havia', 'hei', 'houve', 'houvemos', 'houver', 'houvera', 'houveram',
            'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam',
            'houveríamos', 'houvermos', 'houvesse', 'houvessem', 'houvéramos',
            'houvéssemos', 'há', 'hão', 'isso', 'isto', 'já', 'mais', 'mas',
            'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na',
            'nas', 'nem', 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos',
            'nós', 'num', 'numa', 'não', 'nível', 'ou', 'para', 'pela',
            'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem',
            'são', 'se', 'seja', 'sejam', 'sejamos', 'sem', 'serei', 'seremos',
            'seria', 'seriam', 'seríamos', 'sou', 'sua', 'suas', 'seu', 'seus',
            'só', 'também', 'te', 'tem', 'temos', 'tenha', 'tenham', 'tenhamos',
            'tenho', 'ter', 'terei', 'teremos', 'teria', 'teriam', 'teríamos',
            'teu', 'teus', 'teve', 'tinha', 'tinham', 'tive', 'tivemos',
            'tiver', 'tivera', 'tiveram', 'tiverem', 'tivermos', 'tivesse',
            'tivessem', 'tivéramos', 'tivéssemos', 'tu', 'tua', 'tuas', 'tém',
            'tínhamos', 'um', 'uma', 'você', 'vocês', 'vos', 'à', 'às', 'é'
        ]
        
    def process_vagas_for_dl(self, df_vagas: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados de vagas para Deep Learning
        
        Args:
            df_vagas: DataFrame com dados das vagas
            
        Returns:
            DataFrame normalizado e pronto para Deep Learning
        """
        df = df_vagas.copy()
        
        # Lista para armazenar todas as features processadas
        processed_features = []
        
        # 1. Processamento de dados categóricos
        categorical_cols = [
            'cliente', 'tipo_contratacao', 'analista_responsavel', 
            'estado', 'cidade', 'nivel_profissional', 'nivel_academico',
            'nivel_ingles', 'nivel_espanhol', 'areas_atuacao'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                # Preenche valores ausentes
                df[col] = df[col].fillna('Não informado')
                
                # One-hot encoding para categorias com poucos valores únicos
                if df[col].nunique() <= 20:
                    encoded = pd.get_dummies(df[col], prefix=f'{col}_')
                    processed_features.append(encoded)
                else:
                    # Label encoding para categorias com muitos valores
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col])
                    processed_features.append(pd.DataFrame({f'{col}_encoded': encoded}))
                    self.encoders[f'vagas_{col}'] = le
        
        # 2. Processamento de dados numéricos
        numeric_cols = ['valor_venda_num']
        
        for col in numeric_cols:
            if col in df.columns:
                # Tratamento de valores ausentes
                imputer = SimpleImputer(strategy='median')
                values = imputer.fit_transform(df[[col]])
                
                # Normalização
                scaler = StandardScaler()
                normalized = scaler.fit_transform(values)
                
                processed_features.append(pd.DataFrame({f'{col}_normalized': normalized.flatten()}))
                self.scalers[f'vagas_{col}'] = scaler
                self.imputers[f'vagas_{col}'] = imputer
        
        # 3. Processamento de dados de texto
        text_cols = ['titulo']
        
        for col in text_cols:
            if col in df.columns:
                # Limpa e processa texto
                texts = df[col].fillna('').astype(str)
                texts = texts.apply(self._clean_text)
                
                # TF-IDF para criar features numéricas do texto
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                    )
                    
                    processed_features.append(tfidf_df)
                    self.vectorizers[f'vagas_{col}'] = vectorizer
                except Exception as e:
                    # Se TF-IDF falhar, cria features básicas de texto
                    text_features = pd.DataFrame({
                        f'{col}_length': texts.str.len(),
                        f'{col}_word_count': texts.str.split().str.len(),
                        f'{col}_has_content': (texts.str.len() > 0).astype(int)
                    })
                    processed_features.append(text_features)
        
        # 4. Features derivadas de datas
        date_cols = ['data_requisicao']
        
        for col in date_cols:
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors='coerce')
                
                # Features derivadas
                date_features = pd.DataFrame({
                    f'{col}_year': dates.dt.year,
                    f'{col}_month': dates.dt.month,
                    f'{col}_day': dates.dt.day,
                    f'{col}_weekday': dates.dt.weekday,
                    f'{col}_days_since_2020': (dates - pd.Timestamp('2020-01-01')).dt.days
                })
                
                # Normaliza as features de data
                for date_col in date_features.columns:
                    if date_features[date_col].notna().any():
                        scaler = StandardScaler()
                        normalized = scaler.fit_transform(date_features[[date_col]].fillna(0))
                        date_features[f'{date_col}_normalized'] = normalized.flatten()
                        self.scalers[f'vagas_{date_col}'] = scaler
                
                processed_features.append(date_features.select_dtypes(include=[np.number]))
        
        # 5. Combina todas as features
        if processed_features:
            result_df = pd.concat(processed_features, axis=1)
            result_df.index = df.index
            
            # Adiciona ID da vaga
            result_df['vaga_id'] = df['codigo_vaga']
            
            # Salva nomes das features
            self.feature_names['vagas'] = result_df.columns.tolist()
            
            return result_df
        
        return pd.DataFrame()
    
    def process_prospects_for_dl(self, df_prospects: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados de prospects para Deep Learning
        
        Args:
            df_prospects: DataFrame com dados dos prospects
            
        Returns:
            DataFrame normalizado e pronto para Deep Learning
        """
        df = df_prospects.copy()
        processed_features = []
        
        # 1. Dados categóricos
        categorical_cols = ['situacao', 'recrutador']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Não informado')
                
                # Força one-hot encoding para situacao sempre
                if col == 'situacao':
                    encoded = pd.get_dummies(df[col], prefix=f'{col}_')
                    processed_features.append(encoded)
                elif df[col].nunique() <= 15:
                    encoded = pd.get_dummies(df[col], prefix=f'{col}_')
                    processed_features.append(encoded)
                else:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col])
                    processed_features.append(pd.DataFrame({f'{col}_encoded': encoded}))
                    self.encoders[f'prospects_{col}'] = le
        
        # 2. Processamento de texto - comentários
        if 'comentario' in df.columns:
            comments = df['comentario'].fillna('').astype(str)
            comments = comments.apply(self._clean_text)
            
            # TF-IDF para comentários
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words=self.portuguese_stop_words,
                lowercase=True
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(comments)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'comentario_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )
                
                processed_features.append(tfidf_df)
                self.vectorizers['prospects_comentario'] = vectorizer
            except Exception as e:
                # Se TF-IDF falhar, cria features básicas de texto
                text_features = pd.DataFrame({
                    'comentario_length': comments.str.len(),
                    'comentario_word_count': comments.str.split().str.len(),
                    'comentario_has_content': (comments.str.len() > 0).astype(int)
                })
                processed_features.append(text_features)
        
        # 3. Features de data
        date_cols = ['data_candidatura', 'ultima_atualizacao']
        
        for col in date_cols:
            if col in df.columns:
                dates = pd.to_datetime(df[col], errors='coerce')
                
                date_features = pd.DataFrame({
                    f'{col}_year': dates.dt.year,
                    f'{col}_month': dates.dt.month,
                    f'{col}_day': dates.dt.day,
                    f'{col}_weekday': dates.dt.weekday,
                    f'{col}_days_since_2020': (dates - pd.Timestamp('2020-01-01')).dt.days
                })
                
                for date_col in date_features.columns:
                    if date_features[date_col].notna().any():
                        scaler = StandardScaler()
                        normalized = scaler.fit_transform(date_features[[date_col]].fillna(0))
                        date_features[f'{date_col}_normalized'] = normalized.flatten()
                        self.scalers[f'prospects_{date_col}'] = scaler
                
                processed_features.append(date_features.select_dtypes(include=[np.number]))
        
        # 4. Feature de tempo entre candidatura e atualização
        if 'data_candidatura' in df.columns and 'ultima_atualizacao' in df.columns:
            date_cand = pd.to_datetime(df['data_candidatura'], errors='coerce')
            date_update = pd.to_datetime(df['ultima_atualizacao'], errors='coerce')
            
            time_diff = (date_update - date_cand).dt.days
            time_diff = time_diff.fillna(0)
            
            scaler = StandardScaler()
            time_diff_normalized = scaler.fit_transform(time_diff.values.reshape(-1, 1))
            
            processed_features.append(pd.DataFrame({
                'tempo_ate_atualizacao_normalized': time_diff_normalized.flatten()
            }))
            self.scalers['prospects_tempo_atualizacao'] = scaler
        
        # 5. Combina todas as features
        if processed_features:
            result_df = pd.concat(processed_features, axis=1)
            result_df.index = df.index
            
            # Adiciona IDs
            result_df['vaga_id'] = df['codigo_vaga']
            result_df['candidato_id'] = df['codigo_candidato']
            
            self.feature_names['prospects'] = result_df.columns.tolist()
            
            return result_df
        
        return pd.DataFrame()
    
    def process_candidates_for_dl(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Processa dados de candidatos para Deep Learning
        
        Args:
            df_candidates: DataFrame com dados dos candidatos
            
        Returns:
            DataFrame normalizado e pronto para Deep Learning
        """
        df = df_candidates.copy()
        processed_features = []
        
        # 1. Dados categóricos
        categorical_cols = [
            'nivel_academico', 'nivel_ingles', 'nivel_espanhol',
            'area_atuacao', 'inserido_por', 'titulo_profissional'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Não informado')
                
                if df[col].nunique() <= 20:
                    encoded = pd.get_dummies(df[col], prefix=f'{col}_')
                    processed_features.append(encoded)
                else:
                    le = LabelEncoder()
                    encoded = le.fit_transform(df[col])
                    processed_features.append(pd.DataFrame({f'{col}_encoded': encoded}))
                    self.encoders[f'candidates_{col}'] = le
        
        # 2. Processamento de dados numéricos
        numeric_cols = ['remuneracao_num']
        
        for col in numeric_cols:
            if col in df.columns:
                imputer = SimpleImputer(strategy='median')
                values = imputer.fit_transform(df[[col]])
                
                scaler = StandardScaler()
                normalized = scaler.fit_transform(values)
                
                processed_features.append(pd.DataFrame({f'{col}_normalized': normalized.flatten()}))
                self.scalers[f'candidates_{col}'] = scaler
                self.imputers[f'candidates_{col}'] = imputer
        
        # 3. Processamento de texto
        text_cols = ['objetivo_profissional', 'conhecimentos_tecnicos', 'certificacoes']
        
        for col in text_cols:
            if col in df.columns:
                texts = df[col].fillna('').astype(str)
                texts = texts.apply(self._clean_text)
                
                vectorizer = TfidfVectorizer(
                    max_features=80,
                    stop_words=self.portuguese_stop_words,
                    lowercase=True,
                    ngram_range=(1, 2)
                )
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    tfidf_df = pd.DataFrame(
                        tfidf_matrix.toarray(),
                        columns=[f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                    )
                    
                    processed_features.append(tfidf_df)
                    self.vectorizers[f'candidates_{col}'] = vectorizer
                except Exception as e:
                    # Se TF-IDF falhar, cria features básicas de texto
                    text_features = pd.DataFrame({
                        f'{col}_length': texts.str.len(),
                        f'{col}_word_count': texts.str.split().str.len(),
                        f'{col}_has_content': (texts.str.len() > 0).astype(int)
                    })
                    processed_features.append(text_features)
        
        # 4. Features de data
        if 'data_criacao' in df.columns:
            dates = pd.to_datetime(df['data_criacao'], errors='coerce')
            
            date_features = pd.DataFrame({
                'data_criacao_year': dates.dt.year,
                'data_criacao_month': dates.dt.month,
                'data_criacao_day': dates.dt.day,
                'data_criacao_weekday': dates.dt.weekday,
                'data_criacao_days_since_2020': (dates - pd.Timestamp('2020-01-01')).dt.days
            })
            
            for date_col in date_features.columns:
                if date_features[date_col].notna().any():
                    scaler = StandardScaler()
                    normalized = scaler.fit_transform(date_features[[date_col]].fillna(0))
                    date_features[f'{date_col}_normalized'] = normalized.flatten()
                    self.scalers[f'candidates_{date_col}'] = scaler
            
            processed_features.append(date_features.select_dtypes(include=[np.number]))
        
        # 5. Features derivadas de contato
        contact_features = pd.DataFrame()
        
        if 'email' in df.columns:
            emails = df['email'].fillna('')
            contact_features['tem_email'] = (emails != '').astype(int)
            contact_features['email_domain_comum'] = emails.apply(
                lambda x: 1 if any(domain in x.lower() for domain in ['gmail', 'hotmail', 'yahoo', 'outlook']) else 0
            )
        
        if 'telefone' in df.columns:
            telefones = df['telefone'].fillna('')
            contact_features['tem_telefone'] = (telefones != '').astype(int)
        
        if not contact_features.empty:
            processed_features.append(contact_features)
        
        # 6. Combina todas as features
        if processed_features:
            result_df = pd.concat(processed_features, axis=1)
            result_df.index = df.index
            
            # Adiciona ID do candidato
            result_df['candidato_id'] = df['codigo']
            
            self.feature_names['candidates'] = result_df.columns.tolist()
            
            return result_df
        
        return pd.DataFrame()
    
    def _clean_text(self, text: str) -> str:
        """
        Limpa e normaliza texto
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo e normalizado
        """
        if not isinstance(text, str):
            return ''
        
        # Remove quebras de linha e tabs
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        
        # Remove caracteres especiais mas mantém acentos
        text = re.sub(r'[^\w\sÀ-ÿ]', ' ', text)
        
        # Remove números isolados
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove espaços extras
        text = re.sub(r'\s+', ' ', text)
        
        # Converte para minúsculas
        text = text.lower().strip()
        
        # Remove textos muito curtos
        if len(text) < 3:
            return ''
        
        return text
    
    def get_feature_summary(self) -> Dict:
        """
        Retorna resumo das features criadas
        
        Returns:
            Dicionário com resumo das features por dataset
        """
        summary = {}
        
        for dataset_name, features in self.feature_names.items():
            summary[dataset_name] = {
                'total_features': len(features),
                'feature_names': features[:10],  # Primeiras 10 features
                'encoders_used': len([k for k in self.encoders.keys() if k.startswith(dataset_name)]),
                'scalers_used': len([k for k in self.scalers.keys() if k.startswith(dataset_name)]),
                'vectorizers_used': len([k for k in self.vectorizers.keys() if k.startswith(dataset_name)])
            }
        
        return summary
    
    def save_preprocessing_objects(self, filepath: str):
        """
        Salva objetos de pré-processamento para uso posterior
        
        Args:
            filepath: Caminho para salvar os objetos
        """
        import pickle
        
        objects = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'vectorizers': self.vectorizers,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
    
    def load_preprocessing_objects(self, filepath: str):
        """
        Carrega objetos de pré-processamento salvos
        
        Args:
            filepath: Caminho dos objetos salvos
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            objects = pickle.load(f)
        
        self.scalers = objects['scalers']
        self.encoders = objects['encoders']
        self.vectorizers = objects['vectorizers']
        self.imputers = objects['imputers']
        self.feature_names = objects['feature_names']
    
    def create_training_dataset(self, df_vagas_dl: pd.DataFrame, df_prospects_dl: pd.DataFrame, 
                               df_candidates_dl: pd.DataFrame) -> pd.DataFrame:
        """
        Cria dataset de treinamento combinando vagas, prospects e candidatos
        
        Args:
            df_vagas_dl: DataFrame de vagas processado para DL
            df_prospects_dl: DataFrame de prospects processado para DL
            df_candidates_dl: DataFrame de candidatos processado para DL
            
        Returns:
            DataFrame com features combinadas e target label
        """
        
        # Verifica se prospects está vazio
        if df_prospects_dl.empty:
            return pd.DataFrame()
        
        # Merge prospects com vagas
        training_data = df_prospects_dl.merge(
            df_vagas_dl, 
            on='vaga_id', 
            how='inner',
            suffixes=('_prospect', '_vaga')
        )
        
        # Merge com candidatos
        training_data = training_data.merge(
            df_candidates_dl,
            on='candidato_id',
            how='inner',
            suffixes=('', '_candidato')
        )
        
        # Criar target label baseado na situação do candidato
        # Primeiro, vamos pegar a situação original dos prospects
        situacao_cols = [col for col in training_data.columns if 'situacao_' in col]
        
        # Definir exatamente quais colunas representam contratação
        colunas_contratacao_exatas = [
            'situacao__Contratado pela Decision',
            'situacao__Contratado como Hunting',
            'situacao__Documentação CLT',
            'situacao__Documentação Cooperado',
            'situacao__Documentação PJ',
            'situacao__Proposta Aceita'
        ]
        
        # Filtra apenas as colunas que existem no dataset
        contratacao_cols = [col for col in colunas_contratacao_exatas if col in situacao_cols]
        
        if contratacao_cols:
            # Usa as colunas de contratação encontradas
            training_data['target'] = training_data[contratacao_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)
        elif 'situacao_encoded' in training_data.columns:
            # Se houver situacao_encoded, usar lógica específica
            training_data['target'] = training_data['situacao_encoded'].apply(
                lambda x: 1 if x >= 2 else 0  # Ajustar baseado nos dados reais
            )
        else:
            # Fallback - criar target baseado no texto original da situação
            if 'situacao' in df_prospects_dl.columns:
                # Pegar a situação original antes do one-hot encoding
                original_prospects = df_prospects_dl[['vaga_id', 'candidato_id']].copy()
                
                # Tentar recuperar a situação original
                for idx, row in training_data.iterrows():
                    original_situacao = self._get_original_situacao(row, situacao_cols)
                    training_data.loc[idx, 'target'] = self._create_target_label(original_situacao)
            else:
                training_data['target'] = 0  # Default para não contratado
        
        target_distribution = training_data['target'].value_counts()
        
        # Remove colunas de ID para treinamento
        feature_cols = [col for col in training_data.columns 
                       if col not in ['vaga_id', 'candidato_id', 'target']]
        
        # Preenche valores ausentes
        training_data[feature_cols] = training_data[feature_cols].fillna(0)
        
        return training_data
    
    def _get_original_situacao(self, row, situacao_cols):
        """
        Recupera a situação original a partir das colunas one-hot encoded
        """
        for col in situacao_cols:
            if row[col] == 1:
                # Remove o prefixo 'situacao_' para obter o texto original
                return col.replace('situacao_', '')
        return ''
    
    def _create_target_label(self, situacao_raw: str) -> int:
        """
        Cria label de target baseado na situação do candidato
        
        Args:
            situacao_raw: Situação do candidato (texto original)
            
        Returns:
            1 se contratado, 0 caso contrário
        """
        if not isinstance(situacao_raw, str):
            return 0
        
        situacao_lower = situacao_raw.lower()
        
        # Situações que indicam contratação (baseado na análise dos dados)
        situacoes_positivas = [
            'contratado pela decision',
            'contratado como hunting',
            'documentação clt',
            'documentação cooperado', 
            'documentação pj',
            'proposta aceita'
        ]
        
        # Verifica se alguma situação positiva está presente
        for situacao in situacoes_positivas:
            if situacao in situacao_lower:
                return 1
            if situacao in situacao_lower:
                return 1
        
        return 0
    
    def prepare_features_for_training(self, df_training: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features e target para treinamento
        
        Args:
            df_training: DataFrame de treinamento
            
        Returns:
            Tuple com features (X) e target (y)
        """
        # Separa features e target
        X = df_training.drop(['target'], axis=1)
        y = df_training['target']
        
        # Remove colunas de ID se ainda existirem
        id_cols = [col for col in X.columns if 'id' in col.lower()]
        if id_cols:
            X = X.drop(id_cols, axis=1)
        
        # Garante que todas as features são numéricas
        X = X.select_dtypes(include=[np.number])
        
        return X, y