"""
Streamlit веб-интерфейс для демонстрации системы детекции мошенничества
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from inference import FraudPredictor


# Конфигурация страницы
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .clean-alert {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00cc00;
    }
    .warning-alert {
        background-color: #ffffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffcc00;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Загрузка модели (кэшируется)"""
    model_path = '/usr/src/forte/models/fraud_detection_model.pkl'
    
    if not Path(model_path).exists():
        return None
    
    try:
        return FraudPredictor(model_path)
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.markdown('<div class="main-header">🛡️ Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Загрузка модели
    predictor = load_predictor()
    
    if predictor is None:
        st.error("⚠️ Модель не загружена. Пожалуйста, запустите train.py для обучения модели.")
        st.info("Запустите в терминале: `python train.py`")
        return
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Порог классификации
        threshold = st.slider(
            "Порог классификации",
            min_value=0.0,
            max_value=1.0,
            value=float(predictor.model.threshold),
            step=0.01,
            help="Транзакции с вероятностью выше этого порога будут классифицированы как мошеннические"
        )
        predictor.model.threshold = threshold
        
        st.markdown("---")
        
        # Информация о модели
        st.subheader("📊 Информация о модели")
        st.write(f"**Тип модели:** CatBoost")
        st.write(f"**Признаков:** {len(predictor.model.feature_cols)}")
        st.write(f"**Порог:** {threshold:.3f}")
        
        st.markdown("---")
        
        # Режим работы
        mode = st.radio(
            "Режим работы",
            ["Проверка транзакции", "Пакетная проверка"],
            help="Выберите режим: одна транзакция или загрузка CSV файла"
        )
    
    # Основная область
    if mode == "Проверка транзакции":
        show_single_transaction_mode(predictor)
    else:
        show_batch_mode(predictor)


def show_single_transaction_mode(predictor):
    """Режим проверки одной транзакции"""
    
    st.header("💳 Проверка транзакции")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Основные данные")
        
        amount = st.number_input(
            "Сумма перевода (₸)",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )
        
        destination_type = st.selectbox(
            "Тип получателя",
            ["Известный", "Новый"]
        )
        is_new_destination = 1 if destination_type == "Новый" else 0
        
        client_avg_amount = st.number_input(
            "Средняя сумма переводов клиента (₸)",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )
    
    with col2:
        st.subheader("Время транзакции")
        
        hour = st.slider("Час", 0, 23, 12)
        day_of_week = st.selectbox(
            "День недели",
            ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
        )
        day_of_week_num = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"].index(day_of_week)
    
    with col3:
        st.subheader("Дополнительно")
        
        client_tx_count = st.number_input(
            "Количество прошлых транзакций",
            min_value=0,
            value=10,
            step=1
        )
        
        dest_tx_count = st.number_input(
            "Переводов этому получателю",
            min_value=0,
            value=0 if is_new_destination else 3,
            step=1
        )
    
    # Кнопка проверки
    st.markdown("---")
    
    if st.button("🔍 Проверить транзакцию", type="primary", use_container_width=True):
        
        # Формирование данных транзакции
        transaction_data = {
            'amount': amount,
            'hour': hour,
            'day_of_week': day_of_week_num,
            'is_weekend': 1 if day_of_week_num >= 5 else 0,
            'is_night': 1 if hour >= 23 or hour <= 7 else 0,
            'is_morning': 1 if 6 <= hour <= 12 else 0,
            'is_evening': 1 if 18 <= hour <= 23 else 0,
            'log_amount': np.log1p(amount),
            'is_new_destination': is_new_destination,
            'client_tx_count': client_tx_count,
            'client_avg_amount': client_avg_amount,
            'client_median_amount': client_avg_amount,
            'amount_vs_median': amount / (client_avg_amount + 1),
            'amount_vs_avg': amount / (client_avg_amount + 1),
            'dest_tx_count': dest_tx_count,
            'is_round_amount': 1 if amount % 1000 == 0 else 0,
            'is_round_100': 1 if amount % 100 == 0 else 0,
        }
        
        # Предсказание
        with st.spinner("Анализ транзакции..."):
            result = predictor.predict_single_transaction(transaction_data, explain=True)
        
        # Отображение результата
        st.markdown("---")
        st.header("📊 Результат анализа")
        
        # Вероятность
        fraud_prob = result['fraud_probability']
        
        # Gauge chart для вероятности
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Вероятность мошенничества (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ccffcc'},
                    {'range': [30, 80], 'color': '#ffffcc'},
                    {'range': [80, 100], 'color': '#ffcccc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predictor.model.threshold * 100
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Рекомендация
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Вероятность", f"{fraud_prob:.2%}")
        
        with col2:
            st.metric("Классификация", "МОШЕННИЧЕСТВО" if result['is_fraud'] else "ЧИСТАЯ")
        
        with col3:
            st.metric("Рекомендация", result['recommendation'])
        
        # Алерт в зависимости от рекомендации
        if result['recommendation'] == "БЛОКИРОВАТЬ":
            st.markdown(f"""
            <div class="fraud-alert">
                <h3>⛔ ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА</h3>
                <p>Рекомендуется заблокировать транзакцию и провести дополнительную проверку.</p>
            </div>
            """, unsafe_allow_html=True)
        elif result['recommendation'] == "ПРОВЕРИТЬ":
            st.markdown(f"""
            <div class="warning-alert">
                <h3>⚠️ СРЕДНИЙ РИСК</h3>
                <p>Рекомендуется дополнительная проверка перед выполнением транзакции.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="clean-alert">
                <h3>✅ НИЗКИЙ РИСК</h3>
                <p>Транзакция выглядит легитимной.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Топ факторов
        if 'top_factors' in result and result['top_factors']:
            st.markdown("---")
            st.subheader("🔍 Ключевые факторы решения")
            
            factors_df = pd.DataFrame(result['top_factors'])
            
            # График факторов
            fig = px.bar(
                factors_df,
                x='contribution',
                y='feature',
                orientation='h',
                color='contribution',
                color_continuous_scale=['green', 'yellow', 'red'],
                labels={'contribution': 'Вклад в решение', 'feature': 'Признак'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Таблица факторов
            st.dataframe(
                factors_df[['feature', 'value', 'impact']].rename(columns={
                    'feature': 'Признак',
                    'value': 'Значение',
                    'impact': 'Влияние'
                }),
                use_container_width=True,
                hide_index=True
            )


def show_batch_mode(predictor):
    """Режим пакетной проверки"""
    
    st.header("📁 Пакетная проверка транзакций")
    
    st.info("Загрузите CSV файл с транзакциями для массовой проверки")
    
    uploaded_file = st.file_uploader(
        "Выберите CSV файл",
        type=['csv'],
        help="Файл должен содержать необходимые признаки"
    )
    
    if uploaded_file is not None:
        try:
            # Try cp1251 first (our data encoding), then utf-8, then latin1
            try:
                df = pd.read_csv(uploaded_file, encoding='cp1251', sep=';')
            except (UnicodeDecodeError, pd.errors.ParserError):
                uploaded_file.seek(0)  # Reset file pointer
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except (UnicodeDecodeError, pd.errors.ParserError):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')
            
            st.success(f"✓ Загружено {len(df)} транзакций")
            
            # Предпросмотр
            with st.expander("Предпросмотр данных"):
                st.dataframe(df.head(10))
            
            if st.button("Проверить все транзакции", type="primary"):
                with st.spinner("Анализ транзакций..."):
                    predictions = predictor.predict_batch(df)
                
                # Статистика
                st.markdown("---")
                st.subheader("📊 Результаты анализа")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(predictions)
                fraud_count = predictions['is_fraud'].sum()
                block_count = (predictions['recommendation'] == 'БЛОКИРОВАТЬ').sum()
                check_count = (predictions['recommendation'] == 'ПРОВЕРИТЬ').sum()
                
                with col1:
                    st.metric("Всего транзакций", total)
                
                with col2:
                    st.metric("Мошенничество", fraud_count, delta=f"{fraud_count/total*100:.1f}%")
                
                with col3:
                    st.metric("К блокировке", block_count)
                
                with col4:
                    st.metric("К проверке", check_count)
                
                # Распределение вероятностей
                fig = px.histogram(
                    predictions,
                    x='fraud_probability',
                    nbins=50,
                    title="Распределение вероятностей мошенничества",
                    labels={'fraud_probability': 'Вероятность мошенничества'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Таблица с результатами
                st.subheader("Детальные результаты")
                
                # Фильтр
                filter_option = st.selectbox(
                    "Показать",
                    ["Все", "Только мошенничество", "К блокировке", "К проверке"]
                )
                
                if filter_option == "Только мошенничество":
                    display_df = predictions[predictions['is_fraud'] == 1]
                elif filter_option == "К блокировке":
                    display_df = predictions[predictions['recommendation'] == 'БЛОКИРОВАТЬ']
                elif filter_option == "К проверке":
                    display_df = predictions[predictions['recommendation'] == 'ПРОВЕРИТЬ']
                else:
                    display_df = predictions
                
                st.dataframe(
                    display_df[['fraud_probability', 'is_fraud', 'recommendation']],
                    use_container_width=True
                )
                
                # Скачивание результатов
                csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Скачать результаты (CSV)",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()
