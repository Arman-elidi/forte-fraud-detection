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
import importlib

# Force reload inference module to get latest code
import inference
importlib.reload(inference)
from inference import FraudPredictor


# Конфигурация страницы
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    :root {
        --forte-magenta: #E6007E;      /* Forte */
        --forte-deep-purple: #5A2A83;  /* Forte Solo */
        --forte-noble-green: #2E7D32;  /* Forte Premier */
        --forte-dark-blue: #003366;   /* Forte Business */
        --forte-blue: #0066CC;        /* Forte Corporate */
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--forte-magenta);
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: var(--forte-dark-blue);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .clean-alert {
        background-color: var(--forte-noble-green);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00cc00;
    }
    .warning-alert {
        background-color: var(--forte-deep-purple);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffcc00;
    }
    /* Streamlit button styling */
    .stButton > button {
        background-color: var(--forte-magenta) !important;
        color: white !important;
        border: none;
        border-radius: 0.25rem;
    }
    .stButton > button:hover {
        background-color: #c5006a !important;
    }
</style>
""", unsafe_allow_html=True)


def load_predictor():
    """Загрузка модели"""
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
    st.markdown('<div class="main-header">Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Загрузка модели
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Модель не загружена. Пожалуйста, запустите train.py для обучения модели.")
        st.info("Запустите в терминале: `python train.py`")
        return
    

    # Боковая панель с настройками
    with st.sidebar:
        st.header("Настройки")
        
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
        st.subheader("Информация о модели")
        st.write(f"**Тип модели:** CatBoost")
        st.write(f"**Признаков:** {len(predictor.model.feature_cols)}")
        st.write(f"**Порог:** {threshold:.3f}")
        
        st.markdown("---")
        
        # Управление моделью
        st.subheader("Управление")
        if st.button("Переобучить модель", help="Запустить процесс дообучения на новых данных"):
            with st.spinner("Обучение модели... Это может занять несколько минут."):
                import subprocess
                try:
                    # Run train.py in a subprocess
                    result = subprocess.run(
                        [sys.executable, "train.py"],
                        capture_output=True,
                        text=True,
                        cwd="/usr/src/forte"
                    )
                    if result.returncode == 0:
                        st.success("Модель успешно переобучена!")
                        st.cache_resource.clear() # Clear cache to reload model
                        # Reload predictor
                        predictor = load_predictor()
                    else:
                        st.error("Ошибка при обучении")
                        with st.expander("Подробности ошибки"):
                            st.code(result.stderr)
                except Exception as e:
                    st.error(f"Ошибка запуска: {e}")

        st.markdown("---")
        
        # Добавляем новый режим объединения файлов
        mode = st.radio(
            "Режим работы",
            ["Проверка транзакции", "Пакетная проверка", "История проверок", "Объединить файлы"],
            help="Выберите режим работы"
        )

    
    # Инициализация истории в session_state
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Основная область - обработка выбранного режима
    if mode == "Проверка транзакции":
        show_single_transaction_mode(predictor)
    elif mode == "Пакетная проверка":
        show_batch_mode(predictor)
    elif mode == "История проверок":
        show_history_mode()
    elif mode == "Объединить файлы":
        show_merge_files_mode()



def show_merge_files_mode():
    """Режим объединения двух CSV файлов (транзакции + поведенческие паттерны)"""
    st.header("Объединить файлы")
    st.markdown("Загрузите два CSV‑файла: файл транзакций и файл с поведенческими паттернами.")

    col1, col2 = st.columns(2)
    with col1:
        trans_file = st.file_uploader("Файл транзакций", type=["csv"], key="trans_file")
    with col2:
        beh_file = st.file_uploader("Файл поведенческих паттернов", type=["csv"], key="beh_file")

    if trans_file is not None and beh_file is not None:
        # Функция загрузки с обработкой кодировок и разделителей
        def load_csv_smart(uploaded):
            # Пробуем разные кодировки и разделители
            encodings = ['cp1251', 'utf-8', 'latin1']
            separators = [';', ',']
            
            for enc in encodings:
                for sep in separators:
                    try:
                        uploaded.seek(0)
                        df = pd.read_csv(uploaded, encoding=enc, sep=sep)
                        # Проверка: если только одна колонка, вероятно разделитель не тот
                        if df.shape[1] > 1:
                            return df
                    except Exception:
                        continue
            return None

        df_trans = load_csv_smart(trans_file)
        df_beh = load_csv_smart(beh_file)

        if df_trans is None or df_beh is None:
            st.error("Не удалось прочитать файлы. Проверьте формат (CSV) и кодировку.")
            return

        # Словарь для переименования колонок (из data_loader.py)
        trans_mapping = {
            'Уникальный идентификатор клиента': 'client_id',
            'Дата совершенной транзакции': 'transaction_date',
            'Дата и время совершенной транзакции': 'transaction_datetime',
            'Сумма совершенного перевода': 'amount',
            'Уникальный идентификатор транзакции': 'transaction_id',
            'Зашифрованный идентификатор получателя/destination транзакции': 'destination_id',
            'Размеченные транзакции(переводы), где 1 - мошенническая операция , 0 - чистая': 'is_fraud'
        }
        
        beh_mapping = {
            'Уникальный идентификатор клиента': 'client_id',
            'Дата совершенной транзакции': 'transaction_date',
            'UniqueCustomerID': 'client_id',
            'date': 'transaction_date'
        }

        # Переименование
        df_trans.rename(columns=trans_mapping, inplace=True)
        df_beh.rename(columns=beh_mapping, inplace=True)

        # Проверка наличия client_id
        if 'client_id' not in df_trans.columns:
            st.error(f"В файле транзакций не найдена колонка 'client_id'. Найдены: {list(df_trans.columns)}")
            return
        if 'client_id' not in df_beh.columns:
            st.error(f"В файле паттернов не найдена колонка 'client_id'. Найдены: {list(df_beh.columns)}")
            return

        st.success(f"✓ Загружено {len(df_trans)} транзакций и {len(df_beh)} записей поведенческих паттернов")

        # Объединяем по client_id
        merged = pd.merge(df_trans, df_beh, on='client_id', how='inner')
        st.info(f"Получено {len(merged)} строк после объединения (inner join по `client_id`).")
        st.dataframe(merged.head(1000))
        if len(merged) > 1000:
            st.warning(f"⚠️ Показаны первые 1000 строк из {len(merged)}. Скачайте CSV для просмотра всех данных.")

        # Кнопка скачивания результата
        csv = merged.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="💾 Скачать объединённый CSV",
            data=csv,
            file_name="merged_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Загрузите оба файла, чтобы увидеть результат.")


def show_single_transaction_mode(predictor):
    """Режим проверки одной транзакции"""
    
    st.header("Проверка транзакции")
    
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
    
    if st.button("Проверить транзакцию", type="primary", use_container_width=True):
        
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
            
            # Save to history
            st.session_state.history.insert(0, {
                'time': pd.Timestamp.now().strftime("%H:%M:%S"),
                'amount': amount,
                'prob': result['fraud_probability'],
                'rec': result['recommendation']
            })
        
        # Отображение результата
        st.markdown("---")
        st.header("Результат анализа")
        
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
        st.plotly_chart(fig, width="stretch")
        
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
                <h3>ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА</h3>
                <p>Рекомендуется заблокировать транзакцию и провести дополнительную проверку.</p>
            </div>
            """, unsafe_allow_html=True)
        elif result['recommendation'] == "ПРОВЕРИТЬ":
            st.markdown(f"""
            <div class="warning-alert">
                <h3>СРЕДНИЙ РИСК</h3>
                <p>Рекомендуется дополнительная проверка перед выполнением транзакции.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="clean-alert">
                <h3>НИЗКИЙ РИСК</h3>
                <p>Транзакция выглядит легитимной.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Топ факторов
        if 'top_factors' in result and result['top_factors']:
            st.markdown("---")
            st.subheader("Ключевые факторы решения")
            
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
            st.plotly_chart(fig, width="stretch")
            
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
    
    st.header("Пакетная проверка транзакций")
    
    st.info("Загрузите CSV файл с транзакциями для массовой проверки")
    
    uploaded_file = st.file_uploader(
        "Выберите CSV файл",
        type=['csv'],
        help="Файл должен содержать необходимые признаки"
    )
    
    if uploaded_file is not None:
        try:
            # Smart CSV loading with separator and encoding detection
            def load_csv_smart(uploaded):
                encodings = ['utf-8', 'cp1251', 'latin1', 'windows-1251']
                separators = [',', ';', '\t', '|']
                
                for enc in encodings:
                    for sep in separators:
                        try:
                            uploaded.seek(0)
                            df = pd.read_csv(uploaded, encoding=enc, sep=sep, low_memory=False)
                            # Check if we got more than one column (successful parsing)
                            if df.shape[1] > 1:
                                return df
                        except Exception:
                            continue
                
                # If all attempts failed, try with default settings
                uploaded.seek(0)
                return pd.read_csv(uploaded, low_memory=False)
            
            df = load_csv_smart(uploaded_file)
            
            st.success(f"✓ Загружено {len(df)} транзакций")

            # Очистка от повторных заголовков (распространенная проблема при склейке файлов)
            # Проверяем, есть ли строки, где значение в колонке совпадает с названием колонки
            rows_before = len(df)
            for col in df.columns:
                # Проверяем только строковые колонки или object
                if df[col].dtype == 'object':
                    # Удаляем строки, где значение равно названию колонки (с учетом возможных пробелов)
                    is_header = df[col].astype(str).str.strip() == col.strip()
                    if is_header.any():
                        df = df[~is_header]
            
            if len(df) < rows_before:
                st.warning(f"⚠️ Удалено {rows_before - len(df)} строк, являющихся повторными заголовками.")
            
            # Check if this is raw behavioral data (not processed features)
            # Look for Russian column names from behavioral patterns file
            behavioral_indicators = [
                'Количество разных версий ОС',
                'Количество разных моделей телефона',
                'Модель телефона из самой последней сессии',
                'Версия ОС из самой последней сессии',
                'Количество уникальных логин-сессий'
            ]
            is_behavioral_data = any(
                any(indicator in str(col) for indicator in behavioral_indicators)
                for col in df.columns
            )
            
            # Check if this has required model features
            required_features = ['amount', 'hour', 'day_of_week']
            has_model_features = all(col in df.columns for col in required_features)
            
            if is_behavioral_data and not has_model_features:
                st.error("""
                ❌ **Обнаружены сырые поведенческие данные**
                
                Этот файл содержит поведенческие паттерны клиентов, но не содержит данных о транзакциях.
                
                **Что нужно сделать:**
                1. Перейдите в режим "Объединить файлы"
                2. Загрузите файл транзакций И файл поведенческих паттернов
                3. Скачайте объединённый файл
                4. Загрузите объединённый файл сюда для пакетной проверки
                
                Или используйте уже готовый файл `demo_batch_ready.csv` для тестирования.
                """)
                return
            
            # Предпросмотр
            with st.expander("Предпросмотр данных"):
                st.dataframe(df.head(10))
            
            # Настройки порога
            threshold = st.slider(
                "Порог классификации (Threshold)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                help="Транзакции с вероятностью выше этого порога будут считаться мошенническими."
            )

            if st.button("Проверить все транзакции", type="primary"):
                with st.spinner("Анализ транзакций..."):
                    # Сохраняем в session_state
                    st.session_state.batch_predictions = predictor.predict_batch(df)
            
            # Если есть результаты в session_state
            if 'batch_predictions' in st.session_state:
                predictions = st.session_state.batch_predictions.copy()
                
                # Пересчитываем is_fraud на основе выбранного порога
                predictions['is_fraud'] = (predictions['fraud_probability'] >= threshold).astype(int)
                
                # Статистика
                st.markdown("---")
                st.subheader("Результаты анализа")
                
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
                # Добавляем линию порога
                fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold {threshold}")
                
                st.plotly_chart(fig, width="stretch")
                
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
                    width="stretch"
                )
                
                # Скачивание результатов
                csv = predictions.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты (CSV)",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")


def show_history_mode():
    """Режим просмотра истории"""
    st.header("История проверок")
    
    if not st.session_state.history:
        st.info("История пуста. Проверьте несколько транзакций.")
        return
    
    history_df = pd.DataFrame(st.session_state.history)
    
    # Стилизация таблицы
    def highlight_rec(val):
        color = 'green' if val == 'OK' else 'orange' if val == 'ПРОВЕРИТЬ' else 'red'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(
        history_df.style.map(highlight_rec, subset=['rec']),
        width="stretch"
    )
    
    if st.button("Очистить историю"):
        st.session_state.history = []
        st.rerun()


if __name__ == "__main__":
    main()
