"""
Rule Engine для детекции мошенничества
Дополнительный слой правил поверх ML-модели
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class FraudRuleEngine:
    """
    Rule-based система детекции мошенничества
    Может использоваться как:
    1. Standalone система
    2. Дополнительный слой поверх ML-модели
    """
    
    def __init__(self):
        # Пороги для принятия решений
        self.BLOCK_THRESHOLD = 50
        self.REVIEW_THRESHOLD = 20
        
    def evaluate_transaction(
        self, 
        transaction: Dict,
        client_stats: Dict = None,
        behavioral_stats: Dict = None
    ) -> Tuple[int, str, Dict]:
        """
        Оценка транзакции по правилам
        
        Args:
            transaction: данные транзакции
            client_stats: статистика клиента
            behavioral_stats: поведенческие метрики
            
        Returns:
            (risk_score, decision, triggered_rules)
        """
        risk_score = 0
        triggered_rules = {}
        
        # Извлекаем данные
        amount = transaction.get('amount', 0)
        hour = transaction.get('hour', 12)
        is_new_destination = transaction.get('is_new_destination', 0)
        
        # Статистика клиента (если есть)
        if client_stats:
            client_amount_mean = client_stats.get('amount_mean', amount)
            client_amount_median = client_stats.get('amount_median', amount)
            client_night_ratio = client_stats.get('night_ratio', 0.1)
            client_tx_count = client_stats.get('tx_count', 10)
        else:
            client_amount_mean = amount
            client_amount_median = amount
            client_night_ratio = 0.1
            client_tx_count = 10
        
        # Поведенческие метрики (если есть)
        if behavioral_stats:
            is_outlier_flag = behavioral_stats.get('is_outlier_flag', 0)
            zscore_login = behavioral_stats.get('zscore_avg_login_interval_7d', 0)
            login_freq_spike = behavioral_stats.get('login_frequency_spike', False)
        else:
            is_outlier_flag = 0
            zscore_login = 0
            login_freq_spike = False
        
        # ============================================
        # R1: Аномальный скачок суммы
        # ============================================
        if amount > client_amount_mean * 5:
            points = 30
            risk_score += points
            triggered_rules['R1_amount_spike'] = {
                'points': points,
                'reason': f'Сумма {amount:.0f} в {amount/client_amount_mean:.1f}x раз больше средней {client_amount_mean:.0f}'
            }
        
        # ============================================
        # R2: Velocity (быстрая последовательность)
        # ============================================
        time_since_last = transaction.get('time_since_last_tx', 999)
        if time_since_last < 0.17:  # < 10 минут
            points = 25
            risk_score += points
            triggered_rules['R2_velocity'] = {
                'points': points,
                'reason': f'Транзакция через {time_since_last*60:.0f} минут после предыдущей'
            }
        
        # ============================================
        # R3: Ночной перевод вне паттерна
        # ============================================
        if hour in [0, 1, 2, 3, 4, 5] and client_night_ratio < 0.05:
            points = 20
            risk_score += points
            triggered_rules['R3_night_anomaly'] = {
                'points': points,
                'reason': f'Ночной перевод в {hour}:00, клиент обычно не активен ночью ({client_night_ratio*100:.1f}%)'
            }
        
        # ============================================
        # R4: Новый получатель + крупная сумма
        # ============================================
        if is_new_destination and amount > client_amount_median * 3:
            points = 25
            risk_score += points
            triggered_rules['R4_new_dest_large'] = {
                'points': points,
                'reason': f'Новый получатель + сумма {amount:.0f} в {amount/client_amount_median:.1f}x раз больше медианы'
            }
        
        # ============================================
        # R5: Поведенческая аномалия
        # ============================================
        if is_outlier_flag == 1 or zscore_login > 2:
            points = 20
            risk_score += points
            triggered_rules['R5_behavioral_anomaly'] = {
                'points': points,
                'reason': f'Аномальное поведение логинов (Z-score: {zscore_login:.2f})'
            }
        
        # ============================================
        # R6: Всплеск частоты логинов
        # ============================================
        if login_freq_spike:
            points = 20
            risk_score += points
            triggered_rules['R6_login_spike'] = {
                'points': points,
                'reason': 'Резкое увеличение частоты входов за последние 24 часа'
            }
        
        # ============================================
        # R7: Структурирование (мелкие переводы)
        # ============================================
        is_round = transaction.get('is_round_100', 0)
        if is_round and amount < client_amount_median * 0.5:
            points = 15
            risk_score += points
            triggered_rules['R7_structuring'] = {
                'points': points,
                'reason': f'Подозрение на структурирование: круглая сумма {amount:.0f} меньше обычной'
            }
        
        # ============================================
        # R8: Новый клиент + нестабильное поведение
        # ============================================
        if client_tx_count < 5 and is_outlier_flag == 1:
            points = 15
            risk_score += points
            triggered_rules['R8_new_client_unstable'] = {
                'points': points,
                'reason': f'Новый клиент ({client_tx_count} транзакций) с нестабильным поведением'
            }
        
        # ============================================
        # Принятие решения
        # ============================================
        if risk_score >= self.BLOCK_THRESHOLD:
            decision = "БЛОКИРОВАТЬ"
        elif risk_score >= self.REVIEW_THRESHOLD:
            decision = "ПРОВЕРИТЬ"
        else:
            decision = "OK"
        
        return risk_score, decision, triggered_rules
    
    def evaluate_batch(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Пакетная оценка транзакций
        
        Args:
            transactions_df: DataFrame с транзакциями
            
        Returns:
            DataFrame с добавленными колонками rule_score, rule_decision
        """
        results = []
        
        for idx, row in transactions_df.iterrows():
            # Формируем данные для оценки
            transaction = row.to_dict()
            
            # Простая статистика клиента (можно улучшить)
            client_stats = {
                'amount_mean': row.get('client_avg_amount', row['amount']),
                'amount_median': row.get('client_median_amount', row['amount']),
                'night_ratio': 0.1,  # Можно вычислить из истории
                'tx_count': row.get('client_tx_count', 10)
            }
            
            # Поведенческие метрики
            behavioral_stats = {
                'is_outlier_flag': 0,  # Можно добавить из данных
                'zscore_avg_login_interval_7d': row.get('zscore_avg_login_interval_7d', 0),
                'login_frequency_spike': False
            }
            
            # Оценка
            risk_score, decision, rules = self.evaluate_transaction(
                transaction, client_stats, behavioral_stats
            )
            
            results.append({
                'rule_score': risk_score,
                'rule_decision': decision,
                'triggered_rules_count': len(rules)
            })
        
        # Добавляем результаты
        result_df = pd.DataFrame(results)
        return pd.concat([transactions_df, result_df], axis=1)
    
    def combine_with_ml(
        self, 
        ml_probability: float, 
        rule_score: int,
        ml_weight: float = 0.7
    ) -> Tuple[str, str]:
        """
        Комбинирование ML-модели и Rule Engine
        
        Args:
            ml_probability: вероятность от ML-модели (0-1)
            rule_score: баллы от rule engine (0-100+)
            ml_weight: вес ML-модели (0-1)
            
        Returns:
            (final_decision, explanation)
        """
        # Нормализуем rule_score к [0, 1]
        rule_probability = min(rule_score / 100, 1.0)
        
        # Взвешенная комбинация
        combined_score = ml_weight * ml_probability + (1 - ml_weight) * rule_probability
        
        # Решение
        if combined_score >= 0.8 or rule_score >= self.BLOCK_THRESHOLD:
            decision = "БЛОКИРОВАТЬ"
            explanation = f"Высокий риск: ML={ml_probability:.2%}, Rules={rule_score} баллов"
        elif combined_score >= 0.3 or rule_score >= self.REVIEW_THRESHOLD:
            decision = "ПРОВЕРИТЬ"
            explanation = f"Средний риск: ML={ml_probability:.2%}, Rules={rule_score} баллов"
        else:
            decision = "OK"
            explanation = f"Низкий риск: ML={ml_probability:.2%}, Rules={rule_score} баллов"
        
        return decision, explanation


# Пример использования
if __name__ == "__main__":
    # Инициализация
    rule_engine = FraudRuleEngine()
    
    # Пример 1: Легитимная транзакция
    print("=" * 60)
    print("ПРИМЕР 1: Легитимная транзакция")
    print("=" * 60)
    
    transaction1 = {
        'amount': 5000,
        'hour': 14,
        'is_new_destination': 0,
        'time_since_last_tx': 24,
        'is_round_100': 0
    }
    
    client_stats1 = {
        'amount_mean': 5000,
        'amount_median': 4500,
        'night_ratio': 0.05,
        'tx_count': 25
    }
    
    score, decision, rules = rule_engine.evaluate_transaction(transaction1, client_stats1)
    
    print(f"Risk Score: {score}")
    print(f"Decision: {decision}")
    print(f"Triggered Rules: {len(rules)}")
    for rule_name, rule_info in rules.items():
        print(f"  - {rule_name}: +{rule_info['points']} ({rule_info['reason']})")
    
    # Пример 2: Мошенническая транзакция
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Мошенническая транзакция")
    print("=" * 60)
    
    transaction2 = {
        'amount': 100000,
        'hour': 2,
        'is_new_destination': 1,
        'time_since_last_tx': 0.08,  # 5 минут
        'is_round_100': 1
    }
    
    client_stats2 = {
        'amount_mean': 8000,
        'amount_median': 7000,
        'night_ratio': 0.02,
        'tx_count': 15
    }
    
    behavioral_stats2 = {
        'is_outlier_flag': 1,
        'zscore_avg_login_interval_7d': 3.5,
        'login_frequency_spike': True
    }
    
    score, decision, rules = rule_engine.evaluate_transaction(
        transaction2, client_stats2, behavioral_stats2
    )
    
    print(f"Risk Score: {score}")
    print(f"Decision: {decision}")
    print(f"Triggered Rules: {len(rules)}")
    for rule_name, rule_info in rules.items():
        print(f"  - {rule_name}: +{rule_info['points']} ({rule_info['reason']})")
    
    # Пример 3: Комбинация с ML
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: Комбинация ML + Rule Engine")
    print("=" * 60)
    
    ml_prob = 0.65
    rule_score = 45
    
    final_decision, explanation = rule_engine.combine_with_ml(ml_prob, rule_score)
    print(f"ML Probability: {ml_prob:.2%}")
    print(f"Rule Score: {rule_score}")
    print(f"Final Decision: {final_decision}")
    print(f"Explanation: {explanation}")
