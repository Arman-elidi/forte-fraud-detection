# 🗄️ SQL-запросы для Real-Time Fraud Detection

##  Схема базы данных

### Таблица: transactions
```sql
CREATE TABLE transactions (
    docno VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    dest_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    transdatetime TIMESTAMP NOT NULL,
    is_fraud BOOLEAN DEFAULT FALSE,
    
    -- Индексы для быстрых запросов
    INDEX idx_client_datetime (client_id, transdatetime),
    INDEX idx_dest_datetime (dest_id, transdatetime),
    INDEX idx_datetime (transdatetime)
);
```

### Таблица: behavioral_patterns
```sql
CREATE TABLE behavioral_patterns (
    client_id VARCHAR(50) PRIMARY KEY,
    avg_login_interval DECIMAL(10,2),
    std_login_interval DECIMAL(10,2),
    var_login_interval DECIMAL(10,2),
    coef_variation_login_interval DECIMAL(10,4),
    fano_factor_login_interval DECIMAL(10,4),
    zscore_avg_login_interval_7d DECIMAL(10,4),
    is_outlier_flag BOOLEAN,
    login_frequency_7d INT,
    login_frequency_30d INT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔍 SQL-запросы для онлайн-проверок

### 1. Статистика клиента (аномалия суммы)

```sql
-- Получение статистики по транзакциям клиента
SELECT
    COUNT(*) AS tx_count,
    AVG(amount) AS amount_mean,
    STDDEV_POP(amount) AS amount_std,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS amount_median,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS amount_p95,
    MAX(amount) AS amount_max
FROM transactions
WHERE client_id = :client_id
  AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days';
```

**Использование:**
```python
if amount > amount_mean * 5:
    risk_score += 30  # R1: Аномальный скачок суммы
```

---

### 2. Velocity-проверка (частота транзакций)

```sql
-- Количество транзакций за последние 10 минут
SELECT COUNT(*) AS cnt_txn_10min
FROM transactions
WHERE client_id = :client_id
  AND transdatetime >= :current_time - INTERVAL '10 minutes';
```

**Расширенная версия (с суммами):**
```sql
-- Транзакции за разные временные окна
SELECT
    COUNT(CASE WHEN transdatetime >= :now - INTERVAL '10 minutes' THEN 1 END) AS cnt_10min,
    COUNT(CASE WHEN transdatetime >= :now - INTERVAL '1 hour' THEN 1 END) AS cnt_1hour,
    COUNT(CASE WHEN transdatetime >= :now - INTERVAL '24 hours' THEN 1 END) AS cnt_24hour,
    SUM(CASE WHEN transdatetime >= :now - INTERVAL '10 minutes' THEN amount ELSE 0 END) AS sum_10min,
    SUM(CASE WHEN transdatetime >= :now - INTERVAL '1 hour' THEN amount ELSE 0 END) AS sum_1hour
FROM transactions
WHERE client_id = :client_id;
```

**Использование:**
```python
if cnt_10min >= 5:
    risk_score += 25  # R2: Velocity-атака
```

---

### 3. Проверка нового получателя

```sql
-- Количество предыдущих транзакций этому получателю
SELECT COUNT(*) AS prev_txn_to_dest
FROM transactions
WHERE client_id = :client_id
  AND dest_id = :dest_id
  AND transdatetime < :current_time;
```

**Расширенная версия (с деталями):**
```sql
-- Детальная информация о получателе
SELECT
    COUNT(*) AS tx_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MAX(transdatetime) AS last_tx_datetime,
    EXTRACT(EPOCH FROM (:current_time - MAX(transdatetime))) / 3600 AS hours_since_last_tx
FROM transactions
WHERE client_id = :client_id
  AND dest_id = :dest_id;
```

**Использование:**
```python
is_new_destination = 1 if prev_txn_to_dest == 0 else 0
if is_new_destination and amount > amount_median * 3:
    risk_score += 25  # R4: Новый получатель + крупная сумма
```

---

### 4. Ночная активность клиента

```sql
-- Доля ночных транзакций клиента
SELECT
    COUNT(*) AS total_tx,
    COUNT(CASE WHEN EXTRACT(HOUR FROM transdatetime) BETWEEN 0 AND 5 THEN 1 END) AS night_tx,
    COUNT(CASE WHEN EXTRACT(HOUR FROM transdatetime) BETWEEN 0 AND 5 THEN 1 END)::FLOAT / 
        NULLIF(COUNT(*), 0) AS night_ratio
FROM transactions
WHERE client_id = :client_id
  AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days';
```

**Использование:**
```python
current_hour = datetime.now().hour
if current_hour in [0,1,2,3,4,5] and night_ratio < 0.05:
    risk_score += 20  # R3: Ночная аномалия
```

---

### 5. Поведенческие паттерны

```sql
-- Получение поведенческих метрик клиента
SELECT
    avg_login_interval,
    coef_variation_login_interval,
    fano_factor_login_interval,
    zscore_avg_login_interval_7d,
    is_outlier_flag,
    login_frequency_7d,
    login_frequency_30d
FROM behavioral_patterns
WHERE client_id = :client_id;
```

**Использование:**
```python
if is_outlier_flag == 1 or zscore_avg_login_interval_7d > 2:
    risk_score += 20  # R5: Поведенческая аномалия
```

---

### 6. Риск получателя (мул-счет)

```sql
-- Статистика получателя (возможный мул)
SELECT
    COUNT(DISTINCT client_id) AS unique_senders,
    COUNT(*) AS total_tx,
    SUM(amount) AS total_received,
    AVG(amount) AS avg_amount,
    MAX(transdatetime) AS last_tx
FROM transactions
WHERE dest_id = :dest_id
  AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '7 days';
```

**Использование:**
```python
if unique_senders > 50:
    risk_score += 30  # Подозрение на мул-счет
```

---

## 🚀 Комплексный запрос для онлайн-скоринга

### Вариант 1: Все метрики одним запросом

```sql
WITH client_stats AS (
    -- Статистика клиента
    SELECT
        COUNT(*) AS tx_count,
        AVG(amount) AS amount_mean,
        STDDEV_POP(amount) AS amount_std,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS amount_median,
        MAX(amount) AS amount_max
    FROM transactions
    WHERE client_id = :client_id
      AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days'
),
velocity AS (
    -- Velocity-метрики
    SELECT
        COUNT(CASE WHEN transdatetime >= :now - INTERVAL '10 minutes' THEN 1 END) AS cnt_10min,
        COUNT(CASE WHEN transdatetime >= :now - INTERVAL '1 hour' THEN 1 END) AS cnt_1hour
    FROM transactions
    WHERE client_id = :client_id
),
dest_check AS (
    -- Проверка получателя
    SELECT
        COUNT(*) AS prev_txn_to_dest,
        COALESCE(MAX(transdatetime), '1970-01-01'::TIMESTAMP) AS last_tx_to_dest
    FROM transactions
    WHERE client_id = :client_id
      AND dest_id = :dest_id
),
dest_risk AS (
    -- Риск получателя
    SELECT
        COUNT(DISTINCT client_id) AS dest_unique_senders,
        COUNT(*) AS dest_total_tx
    FROM transactions
    WHERE dest_id = :dest_id
      AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '7 days'
),
night_pattern AS (
    -- Ночная активность
    SELECT
        COUNT(CASE WHEN EXTRACT(HOUR FROM transdatetime) BETWEEN 0 AND 5 THEN 1 END)::FLOAT /
            NULLIF(COUNT(*), 0) AS night_ratio
    FROM transactions
    WHERE client_id = :client_id
      AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days'
),
behavioral AS (
    -- Поведенческие паттерны
    SELECT
        zscore_avg_login_interval_7d,
        is_outlier_flag,
        fano_factor_login_interval,
        coef_variation_login_interval
    FROM behavioral_patterns
    WHERE client_id = :client_id
)

-- Объединение всех метрик
SELECT
    -- Клиентские метрики
    cs.tx_count,
    cs.amount_mean,
    cs.amount_median,
    cs.amount_max,
    
    -- Velocity
    v.cnt_10min,
    v.cnt_1hour,
    
    -- Получатель
    dc.prev_txn_to_dest,
    EXTRACT(EPOCH FROM (:now - dc.last_tx_to_dest)) / 3600 AS hours_since_last_to_dest,
    
    -- Риск получателя
    dr.dest_unique_senders,
    dr.dest_total_tx,
    
    -- Ночная активность
    np.night_ratio,
    
    -- Поведенческие
    b.zscore_avg_login_interval_7d,
    b.is_outlier_flag,
    b.fano_factor_login_interval,
    b.coef_variation_login_interval,
    
    -- Вычисляемые признаки
    :amount / NULLIF(cs.amount_median, 0) AS amount_vs_median,
    CASE WHEN dc.prev_txn_to_dest = 0 THEN 1 ELSE 0 END AS is_new_destination,
    CASE WHEN dr.dest_unique_senders > 50 THEN 1 ELSE 0 END AS dest_is_mule

FROM client_stats cs
CROSS JOIN velocity v
CROSS JOIN dest_check dc
CROSS JOIN dest_risk dr
CROSS JOIN night_pattern np
LEFT JOIN behavioral b ON 1=1;
```

---

## 🔧 Оптимизация для Production

### 1. Материализованные представления (для часто используемых агрегатов)

```sql
-- Создание материализованного представления для статистики клиентов
CREATE MATERIALIZED VIEW mv_client_stats AS
SELECT
    client_id,
    COUNT(*) AS tx_count_30d,
    AVG(amount) AS amount_mean_30d,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS amount_median_30d,
    MAX(amount) AS amount_max_30d,
    COUNT(CASE WHEN EXTRACT(HOUR FROM transdatetime) BETWEEN 0 AND 5 THEN 1 END)::FLOAT /
        NULLIF(COUNT(*), 0) AS night_ratio
FROM transactions
WHERE transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY client_id;

-- Обновление каждые 5 минут
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_client_stats;
```

### 2. Партиционирование таблицы транзакций

```sql
-- Партиционирование по дате для быстрых запросов
CREATE TABLE transactions (
    docno VARCHAR(50),
    client_id VARCHAR(50),
    dest_id VARCHAR(50),
    amount DECIMAL(15,2),
    transdatetime TIMESTAMP NOT NULL,
    is_fraud BOOLEAN DEFAULT FALSE
) PARTITION BY RANGE (transdatetime);

-- Партиции по месяцам
CREATE TABLE transactions_2025_01 PARTITION OF transactions
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE transactions_2025_02 PARTITION OF transactions
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
```

### 3. Индексы для быстрых запросов

```sql
-- Композитные индексы
CREATE INDEX idx_client_datetime ON transactions(client_id, transdatetime DESC);
CREATE INDEX idx_dest_datetime ON transactions(dest_id, transdatetime DESC);
CREATE INDEX idx_client_dest ON transactions(client_id, dest_id);

-- Частичные индексы (только недавние транзакции)
CREATE INDEX idx_recent_tx ON transactions(client_id, transdatetime)
WHERE transdatetime >= CURRENT_TIMESTAMP - INTERVAL '7 days';
```

---

## 💻 Интеграция с Python

### Пример использования в коде

```python
import psycopg2
from typing import Dict

def get_fraud_features(client_id: str, dest_id: str, amount: float) -> Dict:
    """
    Получение всех признаков для проверки мошенничества
    """
    conn = psycopg2.connect("dbname=fortebank user=postgres")
    cur = conn.cursor()
    
    # Комплексный запрос
    query = """
    WITH client_stats AS (
        SELECT
            AVG(amount) AS amount_mean,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS amount_median
        FROM transactions
        WHERE client_id = %s
          AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '30 days'
    ),
    velocity AS (
        SELECT COUNT(*) AS cnt_10min
        FROM transactions
        WHERE client_id = %s
          AND transdatetime >= CURRENT_TIMESTAMP - INTERVAL '10 minutes'
    ),
    dest_check AS (
        SELECT COUNT(*) AS prev_txn_to_dest
        FROM transactions
        WHERE client_id = %s AND dest_id = %s
    )
    SELECT
        cs.amount_mean,
        cs.amount_median,
        v.cnt_10min,
        dc.prev_txn_to_dest
    FROM client_stats cs
    CROSS JOIN velocity v
    CROSS JOIN dest_check dc;
    """
    
    cur.execute(query, (client_id, client_id, client_id, dest_id))
    result = cur.fetchone()
    
    cur.close()
    conn.close()
    
    return {
        'amount_mean': result[0] or amount,
        'amount_median': result[1] or amount,
        'cnt_10min': result[2] or 0,
        'is_new_destination': 1 if result[3] == 0 else 0,
        'amount_vs_median': amount / (result[1] or amount)
    }
```

---

## 📊 Мониторинг производительности

### Запросы для мониторинга

```sql
-- Топ медленных запросов
SELECT
    query,
    mean_exec_time,
    calls,
    total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Размер таблиц
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## 🎯 Рекомендации

1. **Кэширование** — используйте Redis для часто запрашиваемых метрик
2. **Batch updates** — обновляйте behavioral_patterns раз в час, а не в real-time
3. **Connection pooling** — используйте pgBouncer для управления соединениями
4. **Асинхронность** — используйте async/await для параллельных запросов

---

**Готово для production! 🚀**
