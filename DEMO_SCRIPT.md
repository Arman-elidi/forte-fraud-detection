# Video Demo Script (3 Minutes) — С РЕАЛЬНЫМИ ДАННЫМИ

> **ВАЖНО:** Используйте файл `demo_batch_ready.csv` для демонстрации на реальных данных!

## 0:00 - 0:30: Introduction & Problem
**Visual:** Title Slide -> "The Problem" Slide
**Audio:** "Hello, we are [Team Name]. Mobile banking fraud is a growing threat, costing millions and eroding trust. Traditional rules are too slow. Today, we present our AI-powered Fraud Detection System for ForteBank—a real-time solution that stops fraud in milliseconds."

## 0:30 - 1:00: Live Demo - Batch Processing (REAL DATA)
**Visual:** Streamlit App (Batch Mode)
**Audio:** "Let's see it in action with **real transactions** from our dataset.
*   *Action:* Navigate to 'Пакетная проверка' mode
*   *Action:* Upload `demo_batch_ready.csv`
*   *Result:* 10 transactions processed instantly
*   *Audio:* This file contains 5 legitimate and 5 fraudulent transactions from our actual data. Watch how the model classifies them."

## 1:00 - 1:30: Analyzing Results
**Visual:** Results table + Distribution chart
**Audio:** "Look at the results:
*   *Point to clean transactions:* These legitimate transfers (5000-22000 ₸) get low probabilities (< 20%).
*   *Point to fraud:* But these suspicious ones — 100,000 ₸ to new recipients on weekends — get flagged with 80-90% probability.
*   *Point to distribution chart:* The model clearly separates the two groups."

## 1:30 - 2:00: Explainability & SHAP
**Visual:** Streamlit App (Single Transaction Mode) — pick one fraud example
**Audio:** "Why did it flag this transaction? Let's check one manually.
*   *Action:* Enter fraud example data (100,000 ₸, Saturday, new recipient)
*   *Action:* Click 'Check'
*   *Result:* BLOCK recommendation
*   *Audio:* Look at the 'Key Factors'. The model explains: **New Destination** (+35%), **Weekend** (+20%), **Large Amount** (+18%). This transparency is crucial for analysts."

## 2:00 - 2:30: Adaptability & Retraining
**Visual:** Streamlit App (Sidebar -> Retrain Button)
**Audio:** "Fraud evolves, and so does our model. We've built a 'One-Click Retraining' pipeline.
*   *Action:* Point to 'Retrain Model' button.
*   *Audio:* Analysts can trigger updates as soon as new data is available, keeping the system ahead of new threats without downtime."

## 2:30 - 3:00: Business Value & Conclusion
**Visual:** "Business Value" Slide -> "Conclusion" Slide
**Audio:** "Our solution delivers:
*   **Security:** Blocking 96% of fraud (based on test set).
*   **Speed:** Real-time protection (9ms per transaction).
*   **Trust:** Explainable decisions.
*   **Data:** All examples you saw were from our **real dataset** of 13,000+ transactions.
We are ready to deploy and secure ForteBank's future. Thank you."

---

## Статистика по реальным данным

- **Всего транзакций в датасете**: 13,140
- **Мошеннических**: 165 (1.26%)
- **Легитимных**: 12,975 (98.74%)
- **Файл для демо**: `demo_batch_ready.csv` (10 транзакций)

---

## Инструкция для демонстрации

### Вариант 1: Пакетная проверка (РЕКОМЕНДУЕТСЯ)
1. Запустите `streamlit run app.py`
2. Выберите режим "Пакетная проверка"
3. Загрузите файл `demo_batch_ready.csv`
4. Покажите результаты и распределение вероятностей
5. Объясните, что это **реальные данные** из датасета

### Вариант 2: Ручной ввод
Если хотите показать отдельные примеры вручную, используйте эти значения из реального датасета:

**Легитимная транзакция:**
- Сумма: 5,000 ₸
- Час: 14
- День: Четверг (4)
- Новый получатель: Нет
- Средняя сумма клиента: 80,000 ₸

**Мошенническая транзакция:**
- Сумма: 100,000 ₸
- Час: 16
- День: Суббота (5)
- Новый получатель: Да
- Средняя сумма клиента: 84,000 ₸
