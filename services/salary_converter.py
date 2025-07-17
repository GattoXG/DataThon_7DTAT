#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para conversão e normalização de pretensões salariais
Identifica moeda, frequência e converte tudo para valor mensal em reais
"""

import re
import pandas as pd
from typing import Dict, Tuple, Optional

class SalaryConverter:
    """Classe para conversão de salários com identificação de moeda e frequência"""
    
    def __init__(self):
        # Cotações aproximadas (podem ser atualizadas)
        self.exchange_rates = {
            'USD': 5.0,
            'EUR': 5.5,
            'BRL': 1.0
        }
        
        # Padrões para identificação de moedas
        self.currency_patterns = {
            'BRL': [
                r'\br\$', r'\breal', r'\breais', r'\bbrl\b'
            ],
            'USD': [
                r'\busd\b', r'\bus\$', r'\$\s*usd', r'\bdolar', r'\bdollar', 
                r'us\s+dollar', r'american\s+dollar', r'(?<!r)\$', r'\d+\$'
            ],
            'EUR': [
                r'\beur\b', r'\beuro', r'€', r'european\s+currency'
            ]
        }
        
        # Padrões para identificação de frequências
        self.frequency_patterns = {
            'HORA': [
                r'/\s*h\b', r'\bh\b', r'/\s*hora', r'por\s+hora', r'\bhr\b',
                r'valor\s*/?\s*hora', r'taxa\s*hora', r'hourly', r'hour',
                r'\d+\s*/\s*a\s+hora',  # "75/a hora"
                r'\d+\s*a\s+hora',      # "75 a hora"
                r'\d+[\.,]\d+\s+hora',  # "120,00 hora", "170,00 hora"
                r'\d+h\b',              # "115h"
                r'hora\b',              # Qualquer menção à palavra "hora"
                r'/hr\b',               # "120/hr"
                r'\bhr\b',              # "120 hr", "hr"
                r';\s*hr\b',            # "120;HR"
                r'\d+\s+h\b',           # "R$ 120 h"
                r'r\$\s*\d+\s+h\b'      # "R$ 120 h"
            ],
            'DIA': [
                r'/\s*dia', r'por\s+dia', r'\bdiario', r'\bdaily', r'per\s+day'
            ],
            'SEMANA': [
                r'/\s*semana', r'por\s+semana', r'\bsemanal', r'\bweekly', r'per\s+week'
            ],
            'QUINZENA': [
                r'/\s*quinzena', r'por\s+quinzena', r'\bquinzenal', r'biweekly'
            ],
            'MENSAL': [
                r'/\s*mes', r'/\s*mês', r'por\s+mes', r'por\s+mês', r'\bmensal',
                r'\bmensalista', r'\bmonthly', r'per\s+month', r'regime'
            ],
            'ANUAL': [
                r'/\s*ano', r'por\s+ano', r'\banual', r'\byearly', r'\bannually',
                r'per\s+year', r'\byear\b', r'per\s+annum', r'\bannum\b'
            ]
        }
        
        # Multiplicadores para converter para mensal
        self.frequency_multipliers = {
            'HORA': 220,     # 44h/semana * 5 semanas = 220h/mês
            'DIA': 22,       # 22 dias úteis por mês
            'SEMANA': 4.33,  # 52 semanas / 12 meses
            'QUINZENA': 2,   # 2 quinzenas por mês
            'MENSAL': 1,     # já é mensal
            'ANUAL': 1/12    # dividir por 12
        }
    
    def identify_currency(self, salary_text: str) -> str:
        """Identifica a moeda na string de salário"""
        salary_upper = salary_text.upper()
        
        # Verifica BRL primeiro para evitar conflitos com $ em R$
        if 'R$' in salary_upper or re.search(r'\breal\b|\breais\b|\bbrl\b', salary_upper, re.IGNORECASE):
            return 'BRL'
        
        # Depois verifica outras moedas
        for currency, patterns in self.currency_patterns.items():
            if currency == 'BRL':  # Já verificado acima
                continue
            for pattern in patterns:
                if re.search(pattern, salary_upper, re.IGNORECASE):
                    return currency
        
        return 'BRL'  # Default para Real brasileiro
    
    def identify_frequency(self, salary_text: str) -> str:
        """Identifica a frequência na string de salário"""
        salary_upper = salary_text.upper()
        
        # Padrões específicos para hora (mais prioritários)
        hour_specific_patterns = [
            r'\d+[\.,]?\d*\s+hora\b',    # "120,00 hora", "170,00 hora"
            r'\d+h\b',                   # "115h"
            r'/\s*h\b',                  # "/h"
            r'/\s*hr\b',                 # "/hr"
            r'/\s*hora\b',               # "/hora"
            r'por\s+hora\b',             # "por hora"
            r'valor\s*/?\s*hora',        # "valor/hora", "valor hora"
            r'taxa\s*hora',              # "taxa hora"
            r'hourly\b',                 # "hourly"
            r'\d+\s*/\s*a\s+hora',       # "75/a hora"
            r'\d+\s*a\s+hora',           # "75 a hora"
            r'\d+\s+h\b',                # "120 h", "R$ 120 h"
            r';\s*hr\b',                 # "120;HR"
            r'hr\b'                      # "hr" sozinho
        ]
        
        # Verifica primeiro os padrões específicos de hora
        for pattern in hour_specific_patterns:
            if re.search(pattern, salary_upper, re.IGNORECASE):
                return 'HORA'
        
        # Verifica padrões em ordem de prioridade (mais específicos primeiro)
        for frequency, patterns in self.frequency_patterns.items():
            if frequency == 'HORA':  # Já verificado acima
                continue
            for pattern in patterns:
                if re.search(pattern, salary_upper, re.IGNORECASE):
                    return frequency
        
        return 'MENSAL'  # Default para mensal
    
    def extract_numeric_value(self, salary_text: str) -> Optional[float]:
        """Extrai e normaliza o valor numérico da string"""
        # Remove texto descritivo mas preserva números e separadores
        clean_text = re.sub(r'[^0-9.,\-K]', ' ', salary_text.upper())
        
        # Detecta valores com "K" (milhares)
        if 'K' in clean_text:
            k_match = re.search(r'(\d+(?:[.,]\d+)?)K', clean_text)
            if k_match:
                try:
                    valor_str = k_match.group(1).replace(',', '.')
                    return float(valor_str) * 1000
                except ValueError:
                    pass
        
        # Busca todos os números na string
        number_matches = re.findall(r'(\d+(?:[.,]\d{3})*(?:[.,]\d+)?)', clean_text)
        
        if not number_matches:
            return None
        
        # Se há múltiplos números, pega o maior (geralmente o principal)
        valores = []
        for match in number_matches:
            try:
                valor_normalizado = self._normalize_number_format(match)
                if valor_normalizado and valor_normalizado > 0:
                    valores.append(valor_normalizado)
            except ValueError:
                continue
        
        if valores:
            # Retorna o maior valor encontrado
            return max(valores)
        
        return None
    
    def _normalize_number_format(self, number_str: str) -> Optional[float]:
        """Normaliza formato de número para padrão brasileiro"""
        try:
            # REGRA PRINCIPAL: Analisa contexto para determinar se separador é decimal ou milhares
            
            # Verifica padrão com exatamente 2 dígitos finais: X,XX ou X.XX (centavos)
            centavos_match = re.search(r'[,.](\d{2})$', number_str)
            
            # Verifica padrão com 1 dígito final: X,X ou X.X (pode ser decimal)
            um_digito_match = re.search(r'[,.](\d{1})$', number_str)
            
            if centavos_match:
                # Há exatamente 2 dígitos finais - SEMPRE são centavos
                centavos = centavos_match.group(1)
                
                # Remove os centavos e o separador final para processar a parte inteira
                parte_inteira = number_str[:centavos_match.start()]
                
                # Remove todos os separadores da parte inteira (são separadores de milhares)
                parte_inteira_limpa = re.sub(r'[,.]', '', parte_inteira)
                
                # Reconstrói o número no formato correto
                if parte_inteira_limpa:
                    normalized = parte_inteira_limpa + '.' + centavos
                else:
                    normalized = '0.' + centavos
            
            elif um_digito_match:
                # 1 dígito final: decide baseado no contexto
                digito = um_digito_match.group(1)
                parte_inteira = number_str[:um_digito_match.start()]
                
                # Se a parte inteira é pequena (< 1000), provavelmente é decimal
                # Exemplos: "58,8", "123,5", "999,9"
                parte_inteira_limpa = re.sub(r'[,.]', '', parte_inteira)
                
                if parte_inteira_limpa and int(parte_inteira_limpa) < 1000:
                    # Trata como decimal: 58,8 -> 58.8
                    normalized = parte_inteira_limpa + '.' + digito
                else:
                    # Trata como separador de milhares: remove separador
                    normalized = parte_inteira_limpa + digito if parte_inteira_limpa else digito
            
            else:
                # Não há 1 ou 2 dígitos finais - processa baseado nos separadores
                if ',' in number_str and '.' in number_str:
                    # Ambos presentes: remove todos como separadores de milhares
                    normalized = re.sub(r'[,.]', '', number_str)
                
                elif ',' in number_str:
                    # Apenas vírgula: verifica se pode ser decimal ou é separador de milhares
                    parts = number_str.split(',')
                    if len(parts) == 2 and len(parts[0]) <= 4 and len(parts[1]) > 2:
                        # Caso como "1234,567" - vírgula é separador de milhares
                        normalized = number_str.replace(',', '')
                    else:
                        # Outros casos: remove vírgula como separador de milhares
                        normalized = number_str.replace(',', '')
                
                elif '.' in number_str:
                    # Apenas ponto: remove como separador de milhares
                    normalized = number_str.replace('.', '')
                
                else:
                    # Apenas números
                    normalized = number_str
            
            value = float(normalized)
            
            # Validações de sanidade
            if value <= 0:
                return None
            
            # Ajuste MAIS CRITERIOSO para valores muito pequenos
            # Apenas valores menores que 10 (valores únicos como "5", "8") são multiplicados
            if value < 10 and '.' not in normalized and ',' not in number_str:
                value *= 1000
            
            return value
            
        except (ValueError, TypeError):
            return None
    
    def convert_currency(self, value: float, from_currency: str) -> float:
        """Converte valor de moeda estrangeira para reais"""
        if from_currency == 'BRL':
            return value
        
        rate = self.exchange_rates.get(from_currency, 1.0)
        return value * rate
    
    def convert_frequency_to_monthly(self, value: float, frequency: str) -> float:
        """Converte valor de qualquer frequência para mensal"""
        multiplier = self.frequency_multipliers.get(frequency, 1.0)
        return value * multiplier
    
    def process_salary(self, salary_str: str) -> Dict:
        """Processa completamente uma string de salário"""
        if pd.isna(salary_str) or str(salary_str).strip() == '':
            return {
                'original': salary_str,
                'currency': None,
                'frequency': None,
                'raw_value': None,
                'currency_converted_value': None,
                'monthly_value_brl': None,
                'is_valid': False,
                'error': 'Valor vazio ou nulo'
            }
        
        salary_str = str(salary_str).strip()
        
        # Verifica se é um valor explicitamente inválido
        invalid_patterns = [
            r'^(n/?a|combinado|negociar|negociavel|a\s+combinar|sem\s+informacao)$'
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, salary_str, re.IGNORECASE):
                return {
                    'original': salary_str,
                    'currency': None,
                    'frequency': None,
                    'raw_value': None,
                    'currency_converted_value': None,
                    'monthly_value_brl': None,
                    'is_valid': False,
                    'error': 'Valor inválido explícito'
                }
        
        try:
            # Etapa 1: Identificar moeda
            currency = self.identify_currency(salary_str)
            
            # Etapa 2: Identificar frequência
            frequency = self.identify_frequency(salary_str)
            
            # Etapa 3: Extrair valor numérico
            raw_value = self.extract_numeric_value(salary_str)
            
            if raw_value is None:
                return {
                    'original': salary_str,
                    'currency': currency,
                    'frequency': frequency,
                    'raw_value': None,
                    'currency_converted_value': None,
                    'monthly_value_brl': None,
                    'is_valid': False,
                    'error': 'Não foi possível extrair valor numérico'
                }
            
            # Etapa 4: Converter moeda para reais
            currency_converted_value = self.convert_currency(raw_value, currency)
            
            # Etapa 5: Converter frequência para mensal
            monthly_value_brl = self.convert_frequency_to_monthly(currency_converted_value, frequency)
            
            # Validação final de sanidade
            if monthly_value_brl < 100:
                return {
                    'original': salary_str,
                    'currency': currency,
                    'frequency': frequency,
                    'raw_value': raw_value,
                    'currency_converted_value': currency_converted_value,
                    'monthly_value_brl': monthly_value_brl,
                    'is_valid': False,
                    'error': f'Valor mensal muito baixo: R$ {monthly_value_brl:,.2f}'
                }
            
            # REGRA: Valores finais acima de R$ 100.000 são marcados como não convertidos
            if monthly_value_brl > 100000:
                return {
                    'original': salary_str,
                    'currency': currency,
                    'frequency': frequency,
                    'raw_value': raw_value,
                    'currency_converted_value': currency_converted_value,
                    'monthly_value_brl': monthly_value_brl,
                    'is_valid': False,
                    'error': f'Valor mensal muito alto (não convertido): R$ {monthly_value_brl:,.2f}'
                }
            
            return {
                'original': salary_str,
                'currency': currency,
                'frequency': frequency,
                'raw_value': raw_value,
                'currency_converted_value': currency_converted_value,
                'monthly_value_brl': monthly_value_brl,
                'is_valid': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'original': salary_str,
                'currency': None,
                'frequency': None,
                'raw_value': None,
                'currency_converted_value': None,
                'monthly_value_brl': None,
                'is_valid': False,
                'error': f'Erro no processamento: {str(e)}'
            }
    
    def process_salary_dataframe(self, df: pd.DataFrame, salary_column: str) -> pd.DataFrame:
        """Processa uma coluna de salários em um DataFrame"""
        results = df[salary_column].apply(self.process_salary)
        
        # Expande os resultados em colunas separadas
        result_df = pd.json_normalize(results)
        
        # Adiciona prefixo para evitar conflitos
        result_df.columns = [f'salary_{col}' for col in result_df.columns]
        
        # Combina com o DataFrame original
        return pd.concat([df, result_df], axis=1)
