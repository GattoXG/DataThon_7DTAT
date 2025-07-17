#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serviço de Deep Learning para Job Matching
Treina modelos de redes neurais para prever contratações e matching
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class JobMatchingDataset(Dataset):
    """Dataset personalizado para job matching"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class JobMatchingNet(nn.Module):
    """Rede neural para job matching"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3, num_classes: int = 2):
        super(JobMatchingNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Camadas ocultas com dropout e batch normalization
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Camada de saída
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DeepLearningModel:
    """Classe principal para treinamento e uso do modelo de deep learning"""
    
    def __init__(self, hidden_sizes: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = []
        self.is_trained = False
        
        print(f"Usando device: {self.device}")
    
    def prepare_data(self, df_training: pd.DataFrame, target_col: str = 'target', 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """
        Prepara dados para treinamento
        
        Args:
            df_training: DataFrame com features e target
            target_col: Nome da coluna target
            test_size: Proporção para teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Tuple com DataLoaders (X_train, X_val, y_train, y_val)
        """
        # Separa features e target
        X = df_training.drop([target_col], axis=1)
        y = df_training[target_col]
        
        # Remove colunas de ID se existirem
        id_cols = [col for col in X.columns if 'id' in col.lower()]
        if id_cols:
            X = X.drop(id_cols, axis=1)
        
        # Seleciona apenas colunas numéricas
        X = X.select_dtypes(include=[np.number])
        
        # Preenche valores ausentes
        X = X.fillna(0)
        
        # Salva nomes das features
        self.feature_names = X.columns.tolist()
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalização
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Converte para DataLoaders
        train_dataset = JobMatchingDataset(X_train_scaled, y_train.values)
        test_dataset = JobMatchingDataset(X_test_scaled, y_test.values)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader, X_train_scaled, X_test_scaled
    
    def create_model(self, input_size: int):
        """
        Cria o modelo de rede neural
        
        Args:
            input_size: Número de features de entrada
        """
        self.model = JobMatchingNet(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        print(f"Modelo criado com {input_size} features de entrada")
        print(f"Arquitetura: {input_size} -> {' -> '.join(map(str, self.hidden_sizes))} -> 2")
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                   epochs: int = 100, patience: int = 10) -> Dict:
        """
        Treina o modelo
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação
            epochs: Número de épocas
            patience: Paradas antecipadas
            
        Returns:
            Dicionário com histórico de treinamento
        """
        if self.model is None:
            raise ValueError("Modelo não foi criado. Chame create_model() primeiro.")
        
        # Calcula pesos das classes para balanceamento
        # Conta as classes no conjunto de treinamento
        class_counts = torch.bincount(train_loader.dataset.targets)
        total_samples = len(train_loader.dataset)
        
        # Calcula pesos inversamente proporcionais à frequência
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(self.device)
        
        print(f"Contagem de classes: {class_counts}")
        print(f"Pesos das classes: {class_weights}")
        
        # Usa CrossEntropyLoss com pesos balanceados
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8)
        
        best_val_acc = 0
        patience_counter = 0
        
        self.training_history = []
        
        for epoch in range(epochs):
            # Treinamento
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_targets.size(0)
                train_correct += (predicted == batch_targets).sum().item()
            
            # Validação
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_targets.size(0)
                    val_correct += (predicted == batch_targets).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Calcula F1-Score para validação (importante para dados desbalanceados)
            val_f1 = 0.0
            if val_total > 0:
                with torch.no_grad():
                    val_predictions = []
                    val_true = []
                    
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        outputs = self.model(batch_features)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        val_predictions.extend(predicted.cpu().numpy())
                        val_true.extend(batch_targets.cpu().numpy())
                    
                    # Calcula F1-Score
                    from sklearn.metrics import f1_score
                    val_f1 = f1_score(val_true, val_predictions, average='weighted')
            
            # Salva histórico
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc,
                'val_f1': val_f1
            }
            self.training_history.append(epoch_history)
            
            # Scheduler
            scheduler.step(val_acc)
            
            # Early stopping melhorado - mais tolerante
            improvement = val_acc - best_val_acc
            min_improvement = 0.00005  # Melhoria mínima menor para ser mais tolerante
            
            if improvement > min_improvement:
                best_val_acc = val_acc
                patience_counter = 0
                # Salva melhor modelo
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Lógica de parada mais permissiva
            should_stop = False
            
            if patience_counter >= patience:
                # Seja muito mais tolerante no início (até 50 épocas)
                if epoch < 50:
                    patience_counter = patience // 3  # Reseta mais agressivamente
                    print(f"Epoch {epoch + 1}: Início do treinamento, continuando... (patience resetado)")
                elif epoch < 100:
                    # Verifica se ainda há progresso
                    if len(self.training_history) >= 15:
                        recent_vals = [h['val_acc'] for h in self.training_history[-15:]]
                        trend = np.mean(recent_vals[-5:]) - np.mean(recent_vals[:5])
                        
                        # Mais tolerante com tendência positiva
                        if trend > 0.0005:
                            patience_counter = patience // 2
                            print(f"Epoch {epoch + 1}: Tendência positiva detectada, continuando...")
                        else:
                            should_stop = True
                    else:
                        patience_counter = patience // 2  # Ainda no meio do treinamento
                else:
                    should_stop = True
            
            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
                if patience_counter > 0:
                    print(f"  Patience: {patience_counter}/{patience} (Best: {best_val_acc:.4f})")
        
        # Carrega melhor modelo
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.is_trained = True
        
        return {
            'best_val_accuracy': best_val_acc,
            'total_epochs': len(self.training_history),
            'training_history': self.training_history
        }
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            test_loader: DataLoader de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calcula métricas
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # ROC AUC para classe positiva
        probs_positive = [prob[1] for prob in all_probabilities]
        roc_auc = roc_auc_score(all_targets, probs_positive)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições para novos dados
        
        Args:
            X: DataFrame com features
            
        Returns:
            Array com predições
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        # Processa dados da mesma forma que no treinamento
        X_processed = X.copy()
        
        # Remove colunas de ID se existirem
        id_cols = [col for col in X_processed.columns if 'id' in col.lower()]
        if id_cols:
            X_processed = X_processed.drop(id_cols, axis=1)
        
        # Seleciona apenas colunas numéricas
        X_processed = X_processed.select_dtypes(include=[np.number])
        
        # Preenche valores ausentes
        X_processed = X_processed.fillna(0)
        
        # Garante que tem as mesmas features do treinamento
        if self.feature_names:
            missing_cols = set(self.feature_names) - set(X_processed.columns)
            for col in missing_cols:
                X_processed[col] = 0
            X_processed = X_processed[self.feature_names]
        
        # Normaliza
        X_scaled = self.scaler.transform(X_processed)
        
        # Predição
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def save_model(self, filepath: str, evaluation_results: dict = None, training_results: dict = None):
        """
        Salva o modelo treinado junto com resultados de avaliação
        
        Args:
            filepath: Caminho para salvar o modelo
            evaluation_results: Resultados da avaliação (métricas, matriz de confusão, etc.)
            training_results: Resultados do treinamento (histórico, épocas, etc.)
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'evaluation_results': evaluation_results,
            'training_results': training_results,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega um modelo salvo junto com resultados de avaliação
        
        Args:
            filepath: Caminho do modelo salvo
            
        Returns:
            dict: Dicionário com evaluation_results e training_results se disponíveis
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.hidden_sizes = model_data['hidden_sizes']
        self.dropout_rate = model_data['dropout_rate']
        self.learning_rate = model_data['learning_rate']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        # Carrega resultados de avaliação se disponíveis
        evaluation_results = model_data.get('evaluation_results', {})
        training_results = model_data.get('training_results', {})
        save_timestamp = model_data.get('save_timestamp', None)
        
        # Recria o modelo
        if self.feature_names:
            self.create_model(len(self.feature_names))
            self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"Modelo carregado de: {filepath}")
        if save_timestamp:
            print(f"Modelo salvo em: {save_timestamp}")
        
        return {
            'evaluation_results': evaluation_results,
            'training_results': training_results,
            'save_timestamp': save_timestamp
        }
    
    def get_feature_importance(self, X_sample: pd.DataFrame, method: str = 'permutation') -> pd.DataFrame:
        """
        Calcula importância das features
        
        Args:
            X_sample: Amostra de dados para calcular importância
            method: Método para calcular importância
            
        Returns:
            DataFrame com importância das features
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda.")
        
        if method == 'permutation':
            # Implementação simples de permutation importance
            base_predictions, _ = self.predict(X_sample)
            base_accuracy = accuracy_score(
                [1] * len(base_predictions), 
                base_predictions
            )
            
            importances = []
            
            for col in self.feature_names:
                X_permuted = X_sample.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col])
                
                permuted_predictions, _ = self.predict(X_permuted)
                permuted_accuracy = accuracy_score(
                    [1] * len(permuted_predictions),
                    permuted_predictions
                )
                
                importance = base_accuracy - permuted_accuracy
                importances.append(importance)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        
        else:
            raise ValueError(f"Método '{method}' não suportado.")

def create_job_matching_model(df_vagas_dl: pd.DataFrame, df_prospects_dl: pd.DataFrame, 
                            df_candidates_dl: pd.DataFrame, 
                            hidden_sizes: List[int] = [256, 128, 64],
                            epochs: int = 100) -> DeepLearningModel:
    """
    Função auxiliar para criar e treinar um modelo de job matching
    
    Args:
        df_vagas_dl: DataFrame de vagas processado
        df_prospects_dl: DataFrame de prospects processado
        df_candidates_dl: DataFrame de candidatos processado
        hidden_sizes: Tamanhos das camadas ocultas
        epochs: Número de épocas de treinamento
        
    Returns:
        Modelo treinado
    """
    from .feature_engineering import FeatureEngineering
    
    # Inicializa feature engineering
    fe = FeatureEngineering()
    
    # Cria dataset de treinamento
    training_data = fe.create_training_dataset(
        df_vagas_dl, df_prospects_dl, df_candidates_dl
    )
    
    if training_data.empty:
        raise ValueError("Não foi possível criar dataset de treinamento.")
    
    # Inicializa modelo
    model = DeepLearningModel(hidden_sizes=hidden_sizes)
    
    # Prepara dados
    train_loader, test_loader, X_train, X_test = model.prepare_data(training_data)
    
    # Cria modelo
    model.create_model(X_train.shape[1])
    
    # Treina modelo
    training_results = model.train_model(train_loader, test_loader, epochs=epochs)
    
    # Avalia modelo
    evaluation_results = model.evaluate_model(test_loader)
    
    print("=== Resultados do Treinamento ===")
    print(f"Melhor acurácia de validação: {training_results['best_val_accuracy']:.4f}")
    print(f"Épocas treinadas: {training_results['total_epochs']}")
    
    print("\n=== Resultados da Avaliação ===")
    print(f"Acurácia: {evaluation_results['accuracy']:.4f}")
    print(f"Precisão: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
    print(f"ROC AUC: {evaluation_results['roc_auc']:.4f}")
    
    return model
