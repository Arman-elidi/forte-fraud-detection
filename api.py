"""
FastAPI сервер для детекции мошенничества
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from pathlib import Path
import uvicorn

from inference import FraudPredictor


# Модели данных для API
class TransactionRequest(BaseModel):
    """Запрос на проверку транзакции"""
    client_id: str = Field(..., description="ID клиента")
    amount: float = Field(..., description="Сумма перевода", gt=0)
    destination_id: str = Field(..., description="ID получателя")
    hour: int = Field(..., description="Час транзакции (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="День недели (0-6)", ge=0, le=6)
    is_new_destination: bool = Field(False, description="Первый перевод этому получателю?")
    client_avg_amount: Optional[float] = Field(None, description="Средняя сумма переводов клиента")
    
    # Казахстанские нормативы (Закон РК №205-VIII от 30.06.2025)
    is_phone_call_active: Optional[bool] = Field(False, description="Активный звонок (НБ РК 2025)")
    biometric_verified: Optional[bool] = Field(True, description="Биометрия пройдена (АРРФР 07.2025)")
    device_changed_24h: Optional[bool] = Field(False, description="Смена устройства за 24ч (SIM-swap)")
    incoming_transfers_24h: Optional[int] = Field(0, description="Входящих переводов за 24ч (дропперство)")
    time_since_last_incoming: Optional[float] = Field(999, description="Часов с последнего входящего")
    unique_devices_30d: Optional[int] = Field(1, description="Уникальных устройств за 30д (max 10)")
    destination_in_antifraud_db: Optional[bool] = Field(False, description="Получатель в черном списке НБ РК")
    
    # Опциональные поведенческие признаки
    behavioral_features: Optional[Dict[str, float]] = Field(None, description="Поведенческие признаки")


class TransactionResponse(BaseModel):
    """Ответ с результатом проверки"""
    fraud_probability: float = Field(..., description="Вероятность мошенничества (0-1)")
    is_fraud: bool = Field(..., description="Классификация: мошенничество или нет")
    recommendation: str = Field(..., description="Рекомендация: OK / ПРОВЕРИТЬ / БЛОКИРОВАТЬ")
    threshold: float = Field(..., description="Порог классификации")
    top_factors: Optional[List[Dict]] = Field(None, description="Топ факторов, влияющих на решение")


# Создание FastAPI приложения
app = FastAPI(
    title="Fraud Detection API",
    description="API для детекции мошеннических банковских транзакций",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для предиктора
predictor = None


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте сервера"""
    global predictor
    
    model_path = '/usr/src/forte/models/fraud_detection_model.pkl'
    
    if not Path(model_path).exists():
        print(f"⚠️  ВНИМАНИЕ: Модель не найдена по пути {model_path}")
        print("API будет работать, но предсказания будут недоступны")
        print("Пожалуйста, запустите train.py для обучения модели")
        return
    
    try:
        predictor = FraudPredictor(model_path)
        print("✓ Модель успешно загружена")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")


@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=TransactionResponse)
async def predict_transaction(transaction: TransactionRequest, explain: bool = True):
    """
    Предсказание для одной транзакции
    
    Args:
        transaction: данные транзакции
        explain: вычислять ли объяснение (топ факторов)
        
    Returns:
        Результат проверки транзакции
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Пожалуйста, обучите модель (train.py)"
        )
    
    # Преобразуем запрос в словарь признаков
    transaction_data = {
        'amount': transaction.amount,
        'hour': transaction.hour,
        'day_of_week': transaction.day_of_week,
        'is_weekend': 1 if transaction.day_of_week >= 5 else 0,
        'is_night': 1 if transaction.hour >= 23 or transaction.hour <= 7 else 0,
        'is_morning': 1 if 6 <= transaction.hour <= 12 else 0,
        'is_evening': 1 if 18 <= transaction.hour <= 23 else 0,
        'log_amount': __import__('numpy').log1p(transaction.amount),
        'is_new_destination': 1 if transaction.is_new_destination else 0,
        
        # Казахстанские нормативы
        'is_phone_call_active': transaction.is_phone_call_active,
        'biometric_verified': transaction.biometric_verified,
        'device_changed_24h': transaction.device_changed_24h,
        'incoming_transfers_24h': transaction.incoming_transfers_24h,
        'time_since_last_incoming': transaction.time_since_last_incoming,
        'unique_devices_30d': transaction.unique_devices_30d,
        'destination_in_antifraud_db': transaction.destination_in_antifraud_db,
    }
    
    # Добавляем опциональные признаки
    if transaction.client_avg_amount is not None:
        transaction_data['client_avg_amount'] = transaction.client_avg_amount
        transaction_data['amount_vs_avg'] = transaction.amount / (transaction.client_avg_amount + 1)
    
    # Добавляем поведенческие признаки
    if transaction.behavioral_features:
        transaction_data.update(transaction.behavioral_features)
    
    # Предсказание
    try:
        result = predictor.predict_single_transaction(transaction_data, explain=explain)
        return TransactionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Информация о модели"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена"
        )
    
    return {
        "threshold": float(predictor.model.threshold),
        "num_features": len(predictor.model.feature_cols),
        "categorical_features": len(predictor.model.categorical_features),
        "model_type": "CatBoost"
    }


@app.post("/model/set_threshold")
async def set_threshold(threshold: float):
    """
    Изменить порог классификации
    
    Args:
        threshold: новый порог (0-1)
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена"
        )
    
    if not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=400,
            detail="Порог должен быть в диапазоне [0, 1]"
        )
    
    old_threshold = predictor.model.threshold
    predictor.model.threshold = threshold
    
    return {
        "message": "Порог успешно изменён",
        "old_threshold": float(old_threshold),
        "new_threshold": float(threshold)
    }


if __name__ == "__main__":
    print("="*70)
    print("ЗАПУСК FRAUD DETECTION API")
    print("="*70)
    print("\nСервер будет доступен по адресу: http://localhost:8000")
    print("Документация API: http://localhost:8000/docs")
    print("\nДля остановки нажмите Ctrl+C")
    print("="*70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
