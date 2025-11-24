"""
Graph-Based Fraud Detection - Proof of Concept
Демонстрация графового подхода для обнаружения сложных схем мошенничества
"""
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict


class FraudGraphAnalyzer:
    """
    Анализатор графа транзакций для обнаружения сетевых схем мошенничества
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def build_graph(self, transactions_df: pd.DataFrame):
        """
        Построение графа из транзакций
        
        Args:
            transactions_df: DataFrame с транзакциями
        """
        print("Построение графа транзакций...")
        
        for _, row in transactions_df.iterrows():
            client_id = row['client_id']
            dest_id = row['destination_id']
            amount = row['amount']
            datetime = row.get('transaction_datetime', None)
            
            # Добавляем узлы
            self.graph.add_node(client_id, type='client')
            self.graph.add_node(dest_id, type='destination')
            
            # Добавляем ребро (транзакцию)
            if self.graph.has_edge(client_id, dest_id):
                # Обновляем существующее ребро
                self.graph[client_id][dest_id]['count'] += 1
                self.graph[client_id][dest_id]['total_amount'] += amount
            else:
                # Создаем новое ребро
                self.graph.add_edge(
                    client_id, dest_id,
                    count=1,
                    total_amount=amount,
                    last_datetime=datetime
                )
        
        print(f"✓ Граф построен: {self.graph.number_of_nodes()} узлов, {self.graph.number_of_edges()} связей")
    
    def find_money_mules(self, min_senders: int = 20) -> List[Dict]:
        """
        Поиск мул-счетов (получатели от многих отправителей)
        
        Args:
            min_senders: минимальное количество отправителей
            
        Returns:
            Список подозрительных получателей
        """
        print(f"\n🔍 Поиск мул-счетов (получатели от >{min_senders} отправителей)...")
        
        mules = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'destination':
                # Количество уникальных отправителей
                in_degree = self.graph.in_degree(node)
                
                if in_degree >= min_senders:
                    # Общая сумма полученных денег
                    total_received = sum(
                        self.graph[sender][node]['total_amount']
                        for sender in self.graph.predecessors(node)
                    )
                    
                    mules.append({
                        'dest_id': node,
                        'unique_senders': in_degree,
                        'total_received': total_received,
                        'avg_per_sender': total_received / in_degree
                    })
        
        mules.sort(key=lambda x: x['unique_senders'], reverse=True)
        
        print(f"✓ Найдено мул-счетов: {len(mules)}")
        return mules
    
    def find_account_farms(self, min_clients: int = 3) -> List[Dict]:
        """
        Поиск ферм аккаунтов (много клиентов с одинаковыми паттернами)
        
        Упрощенная версия: ищем клиентов, отправляющих одним и тем же получателям
        
        Args:
            min_clients: минимальное количество клиентов в группе
            
        Returns:
            Список подозрительных групп
        """
        print(f"\n🔍 Поиск ферм аккаунтов (группы >{min_clients} клиентов)...")
        
        # Группируем клиентов по общим получателям
        dest_to_clients = defaultdict(set)
        
        for client, dest in self.graph.edges():
            if self.graph.nodes[client].get('type') == 'client':
                dest_to_clients[dest].add(client)
        
        # Ищем группы клиентов с общими получателями
        farms = []
        processed_clients = set()
        
        for dest, clients in dest_to_clients.items():
            if len(clients) >= min_clients:
                # Проверяем, не обработаны ли уже эти клиенты
                if not clients.intersection(processed_clients):
                    farms.append({
                        'common_dest': dest,
                        'clients': list(clients),
                        'client_count': len(clients)
                    })
                    processed_clients.update(clients)
        
        farms.sort(key=lambda x: x['client_count'], reverse=True)
        
        print(f"✓ Найдено ферм: {len(farms)}")
        return farms
    
    def find_transaction_chains(self, max_length: int = 5) -> List[List]:
        """
        Поиск цепочек транзакций (layering)
        
        Args:
            max_length: максимальная длина цепочки
            
        Returns:
            Список цепочек
        """
        print(f"\n🔍 Поиск цепочек транзакций (длина до {max_length})...")
        
        chains = []
        
        # Ищем простые пути в графе
        for source in self.graph.nodes():
            if self.graph.nodes[source].get('type') == 'client':
                for target in self.graph.nodes():
                    if source != target:
                        try:
                            # Все простые пути от source к target
                            paths = list(nx.all_simple_paths(
                                self.graph, source, target, cutoff=max_length
                            ))
                            
                            for path in paths:
                                if len(path) >= 3:  # Минимум 3 узла в цепочке
                                    chains.append(path)
                        except nx.NetworkXNoPath:
                            continue
        
        print(f"✓ Найдено цепочек: {len(chains)}")
        return chains[:50]  # Ограничиваем вывод
    
    def compute_graph_features(self, client_id: str, dest_id: str = None) -> Dict:
        """
        Вычисление графовых признаков для клиента/транзакции
        
        Args:
            client_id: ID клиента
            dest_id: ID получателя (опционально)
            
        Returns:
            Словарь с графовыми признаками
        """
        features = {}
        
        # Признаки клиента
        if client_id in self.graph:
            features['client_out_degree'] = self.graph.out_degree(client_id)  # Количество получателей
            features['client_in_degree'] = self.graph.in_degree(client_id)    # Количество отправителей (если клиент = получатель)
            
            # Clustering coefficient (насколько связаны соседи клиента)
            try:
                features['client_clustering'] = nx.clustering(self.graph.to_undirected(), client_id)
            except:
                features['client_clustering'] = 0
        
        # Признаки получателя
        if dest_id and dest_id in self.graph:
            features['dest_in_degree'] = self.graph.in_degree(dest_id)       # Количество отправителей
            features['dest_out_degree'] = self.graph.out_degree(dest_id)     # Количество получателей (если dest = клиент)
            
            # Является ли хабом (мул-счет)
            features['dest_is_hub'] = 1 if features['dest_in_degree'] > 20 else 0
        
        # Признаки связи
        if client_id in self.graph and dest_id and dest_id in self.graph:
            if self.graph.has_edge(client_id, dest_id):
                edge_data = self.graph[client_id][dest_id]
                features['tx_count_to_dest'] = edge_data['count']
                features['tx_total_to_dest'] = edge_data['total_amount']
            else:
                features['tx_count_to_dest'] = 0
                features['tx_total_to_dest'] = 0
        
        return features
    
    def get_statistics(self) -> Dict:
        """Получение общей статистики графа"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            'density': nx.density(self.graph)
        }


# Пример использования
if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH-BASED FRAUD DETECTION - PROOF OF CONCEPT")
    print("=" * 70)
    
    # Создаем тестовые данные
    print("\n📊 Создание тестовых данных...")
    
    test_data = []
    
    # Обычные клиенты
    for i in range(1, 11):
        test_data.append({
            'client_id': f'client_{i}',
            'destination_id': f'dest_{i}',
            'amount': 5000,
            'transaction_datetime': '2025-01-01'
        })
    
    # Мул-счет (один получатель от многих клиентов)
    for i in range(1, 31):
        test_data.append({
            'client_id': f'client_{i}',
            'destination_id': 'MULE_ACCOUNT',
            'amount': 10000,
            'transaction_datetime': '2025-01-02'
        })
    
    # Ферма аккаунтов (много клиентов отправляют одному получателю)
    for i in range(100, 110):
        test_data.append({
            'client_id': f'farm_client_{i}',
            'destination_id': 'common_dest',
            'amount': 3000,
            'transaction_datetime': '2025-01-03'
        })
    
    # Цепочка (layering)
    test_data.extend([
        {'client_id': 'chain_1', 'destination_id': 'chain_2', 'amount': 50000, 'transaction_datetime': '2025-01-04 10:00'},
        {'client_id': 'chain_2', 'destination_id': 'chain_3', 'amount': 48000, 'transaction_datetime': '2025-01-04 10:15'},
        {'client_id': 'chain_3', 'destination_id': 'chain_4', 'amount': 45000, 'transaction_datetime': '2025-01-04 10:30'},
    ])
    
    df = pd.DataFrame(test_data)
    print(f"✓ Создано {len(df)} тестовых транзакций")
    
    # Инициализация анализатора
    analyzer = FraudGraphAnalyzer()
    
    # Построение графа
    analyzer.build_graph(df)
    
    # Статистика
    stats = analyzer.get_statistics()
    print(f"\n📈 Статистика графа:")
    print(f"  Узлов: {stats['total_nodes']}")
    print(f"  Связей: {stats['total_edges']}")
    print(f"  Средняя степень: {stats['avg_degree']:.2f}")
    print(f"  Плотность: {stats['density']:.4f}")
    
    # Поиск мул-счетов
    mules = analyzer.find_money_mules(min_senders=10)
    if mules:
        print(f"\n🚨 Топ-3 мул-счета:")
        for i, mule in enumerate(mules[:3], 1):
            print(f"  {i}. {mule['dest_id']}: {mule['unique_senders']} отправителей, {mule['total_received']:.0f}₸")
    
    # Поиск ферм
    farms = analyzer.find_account_farms(min_clients=5)
    if farms:
        print(f"\n🚨 Топ-3 фермы аккаунтов:")
        for i, farm in enumerate(farms[:3], 1):
            print(f"  {i}. Общий получатель: {farm['common_dest']}, клиентов: {farm['client_count']}")
    
    # Поиск цепочек
    chains = analyzer.find_transaction_chains(max_length=5)
    if chains:
        print(f"\n🚨 Примеры цепочек транзакций:")
        for i, chain in enumerate(chains[:3], 1):
            print(f"  {i}. {' → '.join(chain)}")
    
    # Вычисление графовых признаков
    print(f"\n📊 Графовые признаки для 'client_1' → 'MULE_ACCOUNT':")
    features = analyzer.compute_graph_features('client_1', 'MULE_ACCOUNT')
    for feature, value in features.items():
        print(f"  {feature}: {value}")
    
    print("\n" + "=" * 70)
    print("✓ Демонстрация завершена!")
    print("=" * 70)
