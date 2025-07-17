#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serviço de Job Matching
Utiliza modelos treinados para fazer matching entre candidatos e vagas
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
from services.deep_learning_model import DeepLearningModel
from services.feature_engineering import FeatureEngineering
from services.data_manager import load_data, process_vagas_for_deep_learning, process_prospects_for_deep_learning, process_candidates_for_deep_learning


class JobMatchingService:
    """Serviço para matching entre candidatos e vagas usando modelos treinados"""
    
    def __init__(self):
        self.model = None
        self.feature_engineer = None
        self.df_vagas = None
        self.df_candidates = None
        self.model_info = {}
        
    def load_model(self, model_path: str) -> Dict:
        """
        Carrega um modelo treinado
        
        Args:
            model_path: Caminho para o modelo
            
        Returns:
            Informações do modelo carregado
        """
        try:
            self.model = DeepLearningModel()
            saved_data = self.model.load_model(model_path)
            
            # Carrega dados originais
            vagas, prospects, applicants = load_data()
            
            # Processa dados para ML (features encodadas)
            self.df_vagas_ml = process_vagas_for_deep_learning(vagas)
            self.df_candidates_ml = process_candidates_for_deep_learning(applicants)
            
            # Processa dados para exibição (informações descritivas)
            self.df_vagas = self._create_vagas_display_data(vagas, prospects)
            self.df_candidates = self._create_candidates_display_data(applicants, prospects)
            
            # Inicializa feature engineering
            self.feature_engineer = FeatureEngineering()
            
            self.model_info = {
                'model_path': model_path,
                'evaluation_results': saved_data.get('evaluation_results', {}),
                'training_results': saved_data.get('training_results', {}),
                'save_timestamp': saved_data.get('save_timestamp', None),
                'is_loaded': True
            }
            
            return self.model_info
            
        except Exception as e:
            raise Exception(f"Erro ao carregar modelo: {str(e)}")
    
    def _create_vagas_display_data(self, vagas, prospects) -> pd.DataFrame:
        """
        Cria DataFrame com informações descritivas das vagas para exibição
        
        Args:
            vagas: Dados das vagas
            prospects: Dados dos prospects
            
        Returns:
            DataFrame com informações descritivas das vagas
        """
        # Identifica vagas que ainda têm candidatos não contratados
        vagas_ativas = set()
        situacoes_contratadas = {'Contratado pela Decision', 'Contratado como Hunting', 'Documentação CLT', 'Documentação Cooperado', 'Documentação PJ'}
        
        for vaga_id, vaga_data in prospects.items():
            tem_candidatos_ativos = False
            for prospect in vaga_data.get('prospects', []):
                situacao = prospect.get('situacao_candidado', '')
                if situacao and situacao not in situacoes_contratadas:
                    tem_candidatos_ativos = True
                    break
            
            if tem_candidatos_ativos:
                vagas_ativas.add(vaga_id)
        
        # Cria DataFrame com informações descritivas
        vagas_display = []
        for vaga_id, vaga_data in vagas.items():
            if vaga_id in vagas_ativas:
                # Extrai informações descritivas seguindo estrutura do data_manager
                info_basicas = vaga_data.get('informacoes_basicas', {})
                perfil_vaga = vaga_data.get('perfil_vaga', {})
                
                # Localização pode estar em localizacao ou perfil_vaga
                localizacao = vaga_data.get('localizacao', {})
                cidade = localizacao.get('cidade', '') or perfil_vaga.get('cidade', '')
                estado = localizacao.get('estado', '') or perfil_vaga.get('estado', '')
                
                vagas_display.append({
                    'vaga_id': vaga_id,
                    'titulo': info_basicas.get('titulo_vaga', f'Vaga {vaga_id}'),
                    'empresa': info_basicas.get('cliente', 'N/A'),
                    'senioridade': perfil_vaga.get('nivel profissional', 'N/A'),
                    'area': perfil_vaga.get('areas_atuacao', 'N/A'),
                    'salario': 0.0,  # Será preenchido após processamento
                    'local': f"{cidade}, {estado}" if cidade and estado else 'N/A'
                })
        
        # Converte para DataFrame
        df_vagas_display = pd.DataFrame(vagas_display)
        
        # Usa o data_manager para processar os valores de salário corretamente
        if not df_vagas_display.empty:
            from services.data_manager import process_vagas_data
            df_vagas_processed = process_vagas_data(vagas)
            
            # Merge com dados processados para obter salários normalizados
            df_merged = df_vagas_display.merge(
                df_vagas_processed[['codigo_vaga', 'valor_venda_num']], 
                left_on='vaga_id', 
                right_on='codigo_vaga', 
                how='left'
            )
            
            # Atualiza salários com valores normalizados (mantém None para valores inválidos)
            df_vagas_display['salario'] = df_merged['valor_venda_num']
            
            # Converte None para 0.0 apenas para exibição
            df_vagas_display['salario'] = df_vagas_display['salario'].fillna(0.0)
        
        return df_vagas_display
    
    def _create_candidates_display_data(self, applicants, prospects) -> pd.DataFrame:
        """
        Cria DataFrame com informações descritivas dos candidatos para exibição
        
        Args:
            applicants: Dados dos candidatos
            prospects: Dados dos prospects
            
        Returns:
            DataFrame com informações descritivas dos candidatos
        """
        # Identifica candidatos que foram contratados
        candidatos_contratados = set()
        situacoes_contratadas = {'Contratado pela Decision', 'Contratado como Hunting', 'Documentação CLT', 'Documentação Cooperado', 'Documentação PJ'}
        
        for vaga_id, vaga_data in prospects.items():
            for prospect in vaga_data.get('prospects', []):
                situacao = prospect.get('situacao_candidado', '')
                candidato_id = prospect.get('codigo', '')
                
                if situacao in situacoes_contratadas and candidato_id:
                    candidatos_contratados.add(candidato_id)
        
        # Cria DataFrame com informações descritivas
        candidatos_display = []
        for candidato_id, candidato_data in applicants.items():
            if candidato_id not in candidatos_contratados:
                # Extrai informações descritivas seguindo estrutura do data_manager
                info_pessoais = candidato_data.get('informacoes_pessoais', {})
                infos_basicas = candidato_data.get('infos_basicas', {})
                info_profissionais = candidato_data.get('informacoes_profissionais', {})
                formacao_idiomas = candidato_data.get('formacao_e_idiomas', {})
                
                # Nome pode estar em informacoes_pessoais ou infos_basicas
                nome = (info_pessoais.get('nome', '') or 
                       infos_basicas.get('nome', '') or 
                       f'Candidato {candidato_id}')
                
                # Área pode estar em informacoes_profissionais como area_atuacao
                area = info_profissionais.get('area_atuacao', 'N/A')
                
                candidatos_display.append({
                    'candidato_id': candidato_id,
                    'nome': nome,
                    'senioridade': formacao_idiomas.get('nivel_academico', 'N/A'),
                    'area': area,
                    'salario_desejado': 0.0  # Será preenchido após processamento
                })
        
        # Converte para DataFrame
        df_candidatos_display = pd.DataFrame(candidatos_display)
        
        # Usa o data_manager para processar os valores de salário corretamente
        if not df_candidatos_display.empty:
            from services.data_manager import process_applicants_data
            df_candidatos_processed = process_applicants_data(applicants)
            
            # Merge com dados processados para obter salários normalizados
            df_merged = df_candidatos_display.merge(
                df_candidatos_processed[['codigo', 'remuneracao_num']], 
                left_on='candidato_id', 
                right_on='codigo', 
                how='left'
            )
            
            # Atualiza salários com valores normalizados (mantém None para valores inválidos)
            df_candidatos_display['salario_desejado'] = df_merged['remuneracao_num']
            
            # Converte None para 0.0 apenas para exibição
            df_candidatos_display['salario_desejado'] = df_candidatos_display['salario_desejado'].fillna(0.0)
        
        return df_candidatos_display
    
    def get_available_models(self) -> List[Dict]:
        """
        Lista todos os modelos disponíveis
        
        Returns:
            Lista de modelos disponíveis
        """
        models_dir = "models"
        if not os.path.exists(models_dir):
            return []
        
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith('.pkl') and file.startswith('job_matching_model_'):
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
    
    def predict_batch_candidates_for_job(self, vaga_id: str, candidate_ids: List[str] = None) -> List[Dict]:
        """
        Prediz probabilidades para múltiplos candidatos em uma vaga em batch (muito mais eficiente)
        
        Args:
            vaga_id: ID da vaga
            candidate_ids: Lista de IDs dos candidatos (None para usar todos disponíveis)
            
        Returns:
            Lista de resultados com candidate_id e match_probability
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Modelo não está carregado ou treinado")
        
        # Se não especificado, usa todos os candidatos disponíveis
        if candidate_ids is None:
            if self.df_candidates is None or self.df_candidates.empty:
                return []
            
            # Verificar se a coluna candidato_id existe
            if 'candidato_id' not in self.df_candidates.columns:
                return []
                
            candidate_ids = self.df_candidates['candidato_id'].tolist()
        
        try:
            # Carrega dados originais
            vagas, prospects, applicants = load_data()
            
            # Cria prospects temporários para todos os candidatos em batch
            temp_prospects = []
            for candidate_id in candidate_ids:
                candidate_data = applicants.get(candidate_id, {})
                infos_basicas = candidate_data.get('infos_basicas', {})
                info_pessoais = candidate_data.get('informacoes_pessoais', {})
                
                # Nome pode estar em infos_basicas ou informacoes_pessoais
                nome = (infos_basicas.get('nome', '') or 
                       info_pessoais.get('nome', '') or 
                       f'Candidato {candidate_id}')
                
                temp_prospects.append({
                    'nome': nome,
                    'codigo': candidate_id,
                    'situacao_candidado': 'Prospect',
                    'data_candidatura': '',
                    'ultima_atualizacao': '',
                    'comentario': '',
                    'recrutador': ''
                })
            
            # Cria estrutura de prospects em batch
            temp_prospect_data = {
                vaga_id: {
                    'titulo': vagas.get(vaga_id, {}).get('informacoes_basicas', {}).get('titulo_vaga', ''),
                    'prospects': temp_prospects
                }
            }
            
            # Processa usando o pipeline de feature engineering
            df_prospects_temp = process_prospects_for_deep_learning(temp_prospect_data)
            
            if df_prospects_temp.empty:
                return []
            
            # Cria dataset completo em batch
            training_data = self.feature_engineer.create_training_dataset(
                self.df_vagas_ml, df_prospects_temp, self.df_candidates_ml
            )
            
            if training_data.empty:
                return []
            
            # Remove coluna target se existir
            features = training_data.drop(['target'], axis=1) if 'target' in training_data.columns else training_data
            
            # Remove colunas de ID mas mantém referência aos candidatos
            candidate_mapping = []
            if 'codigo_candidato' in df_prospects_temp.columns:
                candidate_mapping = df_prospects_temp['codigo_candidato'].tolist()
            
            id_cols = [col for col in features.columns if 'id' in col.lower()]
            if id_cols:
                features = features.drop(id_cols, axis=1)
            
            # Verifica se há features válidas
            if features.empty or features.shape[1] == 0:
                return []
            
            # Faz predição em batch (muito mais eficiente!)
            predictions, probabilities = self.model.predict(features)
            
            # Processa resultados
            results = []
            for i, candidate_id in enumerate(candidate_ids):
                if i < len(probabilities) and len(probabilities[i]) > 1:
                    probability = float(probabilities[i][1])  # Classe "Contratado"
                    results.append({
                        'candidate_id': candidate_id,
                        'match_probability': probability
                    })
            
            return results
            
        except Exception as e:
            return []
    
    def predict_batch_jobs_for_candidate(self, candidate_id: str, vaga_ids: List[str] = None) -> List[Dict]:
        """
        Prediz probabilidades para múltiplas vagas para um candidato em batch
        
        Args:
            candidate_id: ID do candidato
            vaga_ids: Lista de IDs das vagas (None para usar todas disponíveis)
            
        Returns:
            Lista de resultados com vaga_id e match_probability
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Modelo não está carregado ou treinado")
        
        # Se não especificado, usa todas as vagas disponíveis
        if vaga_ids is None:
            if self.df_vagas is None or self.df_vagas.empty:
                return []
            vaga_ids = self.df_vagas['vaga_id'].tolist()
        
        try:
            # Carrega dados originais
            vagas, prospects, applicants = load_data()
            
            # Cria prospects temporários para todas as vagas em batch
            temp_prospect_data = {}
            for vaga_id in vaga_ids:
                vaga_data = vagas.get(vaga_id, {})
                info_basicas = vaga_data.get('informacoes_basicas', {})
                
                candidate_data = applicants.get(candidate_id, {})
                infos_basicas = candidate_data.get('infos_basicas', {})
                info_pessoais = candidate_data.get('informacoes_pessoais', {})
                
                # Nome pode estar em infos_basicas ou informacoes_pessoais
                nome = (infos_basicas.get('nome', '') or 
                       info_pessoais.get('nome', '') or 
                       f'Candidato {candidate_id}')
                
                temp_prospect_data[vaga_id] = {
                    'titulo': info_basicas.get('titulo_vaga', f'Vaga {vaga_id}'),
                    'prospects': [{
                        'nome': nome,
                        'codigo': candidate_id,
                        'situacao_candidado': 'Prospect',
                        'data_candidatura': '',
                        'ultima_atualizacao': '',
                        'comentario': '',
                        'recrutador': ''
                    }]
                }
            
            # Processa usando o pipeline de feature engineering
            df_prospects_temp = process_prospects_for_deep_learning(temp_prospect_data)
            
            if df_prospects_temp.empty:
                return []
            
            # Cria dataset completo em batch
            training_data = self.feature_engineer.create_training_dataset(
                self.df_vagas_ml, df_prospects_temp, self.df_candidates_ml
            )
            
            if training_data.empty:
                return []
            
            # Remove coluna target se existir
            features = training_data.drop(['target'], axis=1) if 'target' in training_data.columns else training_data
            
            # Remove colunas de ID
            id_cols = [col for col in features.columns if 'id' in col.lower()]
            if id_cols:
                features = features.drop(id_cols, axis=1)
            
            # Verifica se há features válidas
            if features.empty or features.shape[1] == 0:
                return []
            
            # Faz predição em batch
            predictions, probabilities = self.model.predict(features)
            
            # Processa resultados
            results = []
            for i, vaga_id in enumerate(vaga_ids):
                if i < len(probabilities) and len(probabilities[i]) > 1:
                    probability = float(probabilities[i][1])  # Classe "Contratado"
                    results.append({
                        'vaga_id': vaga_id,
                        'match_probability': probability
                    })
            
            return results
            
        except Exception as e:
            return []
    def find_best_jobs_for_candidate(self, candidate_id: str, top_n: int = 10, min_probability: float = 0.0) -> List[Dict]:
        """
        Encontra as melhores vagas para um candidato usando predições em batch (otimizado)
        
        Args:
            candidate_id: ID do candidato
            top_n: Número de vagas a retornar
            min_probability: Probabilidade mínima para incluir na lista
            
        Returns:
            Lista de vagas ordenadas por probabilidade de match
        """
        if self.df_vagas is None or self.df_vagas.empty:
            raise ValueError("Dados de vagas não carregados")
        
        # Faz predições em batch (muito mais eficiente!)
        batch_results = self.predict_batch_jobs_for_candidate(candidate_id)
        
        if not batch_results:
            return []
        
        # Combina resultados com informações das vagas
        results = []
        successful_predictions = 0
        
        for batch_result in batch_results:
            vaga_id = batch_result['vaga_id']
            probability = batch_result['match_probability']
            
            if probability >= min_probability:
                # Busca informações da vaga
                vaga_info = self.df_vagas[self.df_vagas['vaga_id'] == vaga_id]
                if not vaga_info.empty:
                    results.append({
                        'candidate_id': candidate_id,  # Adiciona candidate_id para filtro
                        'vaga_id': vaga_id,
                        'match_probability': probability,
                        'vaga_info': vaga_info.iloc[0].to_dict()
                    })
                    successful_predictions += 1
        
        # Filtra por compatibilidade de senioridade
        filtered_results = self._filter_by_compatibility(results, 'candidate_to_job')
        
        # Ordena por probabilidade e retorna top_n
        filtered_results.sort(key=lambda x: x['match_probability'], reverse=True)
        return filtered_results[:top_n]
    
    def find_best_candidates_for_job(self, vaga_id: str, top_n: int = 10, min_probability: float = 0.0) -> List[Dict]:
        """
        Encontra os melhores candidatos para uma vaga usando predições em batch (otimizado)
        
        Args:
            vaga_id: ID da vaga
            top_n: Número de candidatos a retornar
            min_probability: Probabilidade mínima para incluir na lista
            
        Returns:
            Lista de candidatos ordenados por probabilidade de match
        """
        if self.df_candidates is None or self.df_candidates.empty:
            raise ValueError("Dados de candidatos não carregados")
        
        # Faz predições em batch (muito mais eficiente!)
        batch_results = self.predict_batch_candidates_for_job(vaga_id)
        
        if not batch_results:
            return []
        
        # Combina resultados com informações dos candidatos
        results = []
        successful_predictions = 0
        
        for batch_result in batch_results:
            candidate_id = batch_result['candidate_id']
            probability = batch_result['match_probability']
            
            if probability >= min_probability:
                # Busca informações do candidato
                candidate_info = self.df_candidates[self.df_candidates['candidato_id'] == candidate_id]
                if not candidate_info.empty:
                    results.append({
                        'vaga_id': vaga_id,  # Adiciona vaga_id para filtro
                        'candidate_id': candidate_id,
                        'candidato_id': candidate_id,  # Mantém ambos para compatibilidade
                        'match_probability': probability,
                        'candidate_info': candidate_info.iloc[0].to_dict()
                    })
                    successful_predictions += 1
        
        # Filtra por compatibilidade de senioridade
        filtered_results = self._filter_by_compatibility(results, 'job_to_candidate')
        
        # Ordena por probabilidade e retorna top_n
        filtered_results.sort(key=lambda x: x['match_probability'], reverse=True)
        return filtered_results[:top_n]
    
    def _is_compatible_seniority(self, candidate_seniority: str, job_seniority: str) -> bool:
        """
        Verifica se a senioridade do candidato é compatível com a vaga
        
        Args:
            candidate_seniority: Senioridade do candidato
            job_seniority: Senioridade da vaga
            
        Returns:
            True se compatível, False caso contrário
        """
        # Normaliza strings
        candidate_seniority = str(candidate_seniority).lower().strip()
        job_seniority = str(job_seniority).lower().strip()
        
        # Mapeamento de senioridade mais rigoroso
        seniority_levels = {
            'assistente': ['assistente', 'assist.', 'assist', 'trainee', 'estagiario', 'estagiário'],
            'junior': ['junior', 'jr', 'júnior'],
            'pleno': ['pleno', 'pl', 'intermediario', 'intermediário', 'analista'],
            'senior': ['senior', 'sr', 'sênior', 'especialista', 'coordenador', 'lider', 'líder', 'supervisor']
        }
        
        def get_seniority_level(seniority_text):
            seniority_text = seniority_text.lower()
            for level, keywords in seniority_levels.items():
                if any(keyword in seniority_text for keyword in keywords):
                    return level
            return 'pleno'  # Default para casos ambíguos
        
        candidate_level = get_seniority_level(candidate_seniority)
        job_level = get_seniority_level(job_seniority)
        
        # Regras de compatibilidade mais restritivas
        compatibility_rules = {
            'assistente': ['assistente', 'junior'],  # Assistente pode ir para assistente ou júnior
            'junior': ['junior', 'pleno'],  # Junior pode ir para júnior ou pleno
            'pleno': ['pleno', 'senior'],  # Pleno pode ir para pleno ou senior
            'senior': ['senior']  # Senior só para senior (não pode "regredir")
        }
        
        # Candidato com pós-graduação deve ser tratado como no mínimo pleno
        if 'pós' in candidate_seniority or 'pos' in candidate_seniority or 'mestrado' in candidate_seniority or 'doutorado' in candidate_seniority:
            candidate_level = 'pleno' if candidate_level in ['assistente', 'junior'] else candidate_level
        
        return job_level in compatibility_rules.get(candidate_level, ['pleno'])
    
    def _filter_by_compatibility(self, matches: List[Dict], match_type: str = 'candidate_to_job') -> List[Dict]:
        """
        Filtra matches por compatibilidade de senioridade e salário
        
        Args:
            matches: Lista de matches
            match_type: Tipo de match ('candidate_to_job' ou 'job_to_candidate')
            
        Returns:
            Lista filtrada de matches
        """
        filtered_matches = []
        
        for match in matches:
            if match_type == 'candidate_to_job':
                # Candidato → Vaga
                candidate_id = match.get('candidate_id')
                vaga_info = match.get('vaga_info', {})
                
                if candidate_id and self.df_candidates is not None:
                    candidate_info = self.df_candidates[self.df_candidates['candidato_id'] == candidate_id]
                    if not candidate_info.empty:
                        candidate_seniority = candidate_info.iloc[0].get('senioridade', '')
                        job_seniority = vaga_info.get('senioridade', '')
                        
                        # Verifica compatibilidade de senioridade
                        if self._is_compatible_seniority(candidate_seniority, job_seniority):
                            # Verifica compatibilidade de salário
                            if self._is_compatible_salary(candidate_info.iloc[0], vaga_info):
                                filtered_matches.append(match)
            
            else:
                # Vaga → Candidato
                vaga_id = match.get('vaga_id')
                candidate_info = match.get('candidate_info', {})
                
                if vaga_id and self.df_vagas is not None:
                    vaga_info = self.df_vagas[self.df_vagas['vaga_id'] == vaga_id]
                    if not vaga_info.empty:
                        job_seniority = vaga_info.iloc[0].get('senioridade', '')
                        candidate_seniority = candidate_info.get('senioridade', '')
                        
                        # Verifica compatibilidade de senioridade
                        if self._is_compatible_seniority(candidate_seniority, job_seniority):
                            # Verifica compatibilidade de salário
                            if self._is_compatible_salary(candidate_info, vaga_info.iloc[0]):
                                filtered_matches.append(match)
        
        return filtered_matches
    
    def _is_compatible_salary(self, candidate_info: Dict, vaga_info: Dict) -> bool:
        """
        Verifica se o salário da vaga é compatível com a expectativa do candidato
        
        Args:
            candidate_info: Informações do candidato
            vaga_info: Informações da vaga
            
        Returns:
            True se compatível, False caso contrário
        """
        # Obtém salário da vaga
        job_salary = vaga_info.get('salario', 0)
        if isinstance(job_salary, (int, float)):
            job_salary = float(job_salary)
        else:
            job_salary = 0.0
        
        # Obtém salário desejado do candidato
        candidate_salary = candidate_info.get('salario_desejado', 0)
        if isinstance(candidate_salary, (int, float)):
            candidate_salary = float(candidate_salary)
        else:
            candidate_salary = 0.0
        
        # REGRA 1: Se qualquer um dos salários for 0 ou não informado, SEMPRE considera compatível
        # Isso permite matches entre vagas/candidatos com e sem informação salarial
        if job_salary <= 0 or candidate_salary <= 0:
            return True
        
        # REGRA 2: Se ambos têm salário informado, verifica compatibilidade com tolerância
        # Margem de tolerância: vaga pode pagar até 20% menos que o desejado
        # ou até 50% mais (para não limitar oportunidades)
        tolerance_min = 0.8  # 20% menos
        tolerance_max = 1.5  # 50% mais
        
        min_acceptable = candidate_salary * tolerance_min
        max_acceptable = candidate_salary * tolerance_max
        
        return min_acceptable <= job_salary <= max_acceptable

    def get_model_info(self) -> Dict:
        """Retorna informações do modelo atual"""
        return self.model_info
    
    def get_available_candidates(self) -> pd.DataFrame:
        """Retorna lista de candidatos disponíveis"""
        if self.df_candidates is None:
            return pd.DataFrame()
        return self.df_candidates
    
    def get_available_jobs(self) -> pd.DataFrame:
        """Retorna lista de vagas disponíveis"""
        if self.df_vagas is None:
            return pd.DataFrame()
        return self.df_vagas
    
    def get_candidate_info(self, candidate_id: str) -> Dict:
        """Retorna informações detalhadas de um candidato"""
        if self.df_candidates is None or self.df_candidates.empty:
            return {}
        
        candidate = self.df_candidates[self.df_candidates.get('candidato_id') == candidate_id]
        if candidate.empty:
            return {}
        
        return candidate.iloc[0].to_dict()
    
    def get_job_info(self, vaga_id: str) -> Dict:
        """Retorna informações detalhadas de uma vaga"""
        if self.df_vagas is None or self.df_vagas.empty:
            return {}
        
        vaga = self.df_vagas[self.df_vagas.get('vaga_id') == vaga_id]
        if vaga.empty:
            return {}
        
        return vaga.iloc[0].to_dict()
    
    def predict_match_probability(self, candidate_id: str, vaga_id: str) -> float:
        """
        Prediz a probabilidade de match entre candidato e vaga (mantido para compatibilidade)
        Para múltiplas predições, use as funções batch que são mais eficientes
        
        Args:
            candidate_id: ID do candidato
            vaga_id: ID da vaga
            
        Returns:
            Probabilidade de match (0-1)
        """
        # Usa a função batch otimizada para uma única predição
        batch_results = self.predict_batch_jobs_for_candidate(candidate_id, [vaga_id])
        
        if batch_results and len(batch_results) > 0:
            return batch_results[0]['match_probability']
        else:
            return 0.0
