"""
REST API для прогнозирования VIEWS рекламных объявлений
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from waitress import serve
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех доменов

# Загрузка модели и энкодеров при старте
print("Загрузка модели...")
try:
    # Пытаемся загрузить лучшую модель
    model = None
    model_name = None
    for name in ['xgb', 'lgb', 'cb']:
        try:
            model = joblib.load(f'model_{name}.pkl')
            model_name = name
            print(f"Модель загружена: {name.upper()}")
            break
        except FileNotFoundError:
            continue
    
    if model is None:
        raise FileNotFoundError("Модель не найдена. Сначала запустите train_model.py")
    
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print(f"Энкодеры и признаки загружены")
    print(f"Количество признаков: {len(feature_names)}")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None
    label_encoders = None
    feature_names = None

def preprocess_input(cpm, channel_name, date_str, label_encoders, feature_names):
    """
    Предобработка входных данных для предсказания
    """
    # Создание DataFrame
    df = pd.DataFrame({
        'CPM': [cpm],
        'CHANNEL_NAME': [channel_name],
        'DATE': [date_str]
    })
    
    # Обработка дат
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    df['day_of_week'] = df['DATE'].dt.dayofweek
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['week_of_year'] = df['DATE'].dt.isocalendar().week
    df['is_weekend'] = (df['DATE'].dt.dayofweek >= 5).astype(int)
    df['quarter'] = df['DATE'].dt.quarter
    df['is_month_start'] = df['DATE'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['DATE'].dt.is_month_end.astype(int)
    
    # Кодирование канала
    if 'channel' in label_encoders:
        le_channel = label_encoders['channel']
        known_channels = set(le_channel.classes_)
        if channel_name in known_channels:
            df['channel_encoded'] = le_channel.transform([channel_name])[0]
        else:
            df['channel_encoded'] = -1  # Неизвестный канал
    else:
        df['channel_encoded'] = 0
    
    # Статистики по каналу
    if 'channel_stats' in label_encoders:
        channel_stats = label_encoders['channel_stats']
        channel_data = channel_stats[channel_stats['CHANNEL_NAME'] == channel_name]
        
        if not channel_data.empty:
            df['channel_avg_views'] = float(channel_data['channel_avg_views'].iloc[0])
            df['channel_median_views'] = float(channel_data['channel_median_views'].iloc[0])
            df['channel_std_views'] = float(channel_data['channel_std_views'].iloc[0])
            df['channel_count'] = float(channel_data['channel_count'].iloc[0])
            df['channel_avg_cpm'] = float(channel_data['channel_avg_cpm'].iloc[0])
            
            # Дополнительные статистики (если доступны)
            if 'channel_avg_clicks' in channel_data.columns:
                df['channel_avg_clicks'] = float(channel_data['channel_avg_clicks'].iloc[0])
            else:
                df['channel_avg_clicks'] = 0.0
            
            if 'channel_avg_actions' in channel_data.columns:
                df['channel_avg_actions'] = float(channel_data['channel_avg_actions'].iloc[0])
            else:
                df['channel_avg_actions'] = 0.0
        else:
            # Неизвестный канал - используем глобальные средние
            df['channel_avg_views'] = label_encoders.get('global_avg_views', 100)
            df['channel_median_views'] = label_encoders.get('global_median_views', 50)
            df['channel_std_views'] = label_encoders.get('global_std_views', 50)
            df['channel_count'] = 1
            df['channel_avg_cpm'] = cpm
            df['channel_avg_clicks'] = 0
            df['channel_avg_actions'] = 0
    else:
        # Если статистик нет
        df['channel_avg_views'] = label_encoders.get('global_avg_views', 100)
        df['channel_median_views'] = label_encoders.get('global_median_views', 50)
        df['channel_std_views'] = label_encoders.get('global_std_views', 50)
        df['channel_count'] = 1
        df['channel_avg_cpm'] = cpm
        df['channel_avg_clicks'] = 0
        df['channel_avg_actions'] = 0
    
    # Убеждаемся, что все необходимые признаки созданы ДО создания ctr/cvr
    # Сначала создаем все базовые признаки, если они отсутствуют
    required_base_features = ['channel_avg_clicks', 'channel_avg_actions']
    for feature in required_base_features:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Дополнительные признаки (ctr, cvr) - используем средние значения по каналу
    # Для предсказания мы не знаем реальные CLICKS и ACTIONS, поэтому используем статистики
    clicks_val = float(df['channel_avg_clicks'].iloc[0]) if 'channel_avg_clicks' in df.columns and len(df) > 0 else 0.0
    actions_val = float(df['channel_avg_actions'].iloc[0]) if 'channel_avg_actions' in df.columns and len(df) > 0 else 0.0
    views_val = float(df['channel_avg_views'].iloc[0]) if 'channel_avg_views' in df.columns and len(df) > 0 else 1.0
    
    # CTR = средние клики / средние просмотры (из статистики канала)
    df['ctr'] = clicks_val / (views_val + 1) if views_val > 0 else 0.0
    
    # CVR = средние действия / средние просмотры (из статистики канала)
    df['cvr'] = actions_val / (views_val + 1) if views_val > 0 else 0.0
    
    # Убеждаемся, что все необходимые признаки из feature_names созданы
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Проверка перед выборкой признаков
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        # Если все еще есть отсутствующие признаки, создаем их
        for feature in missing_features:
            df[feature] = 0.0
    
    # Выбор признаков в правильном порядке
    X = df[feature_names].copy()
    
    # Заполнение пропусков
    X = X.fillna(0)
    
    # Убеждаемся, что все значения числовые
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности API"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Модель не загружена'}), 500
    return jsonify({
        'status': 'ok',
        'model': model_name.upper() if model_name else 'unknown',
        'features_count': len(feature_names) if feature_names else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint для прогнозирования VIEWS
    
    Принимает JSON:
    {
        "cpm": float,
        "channel": str,
        "date": str (формат: "YYYY-MM-DD")
    }
    
    Возвращает:
    {
        "predicted_views": int
    }
    """
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    try:
        # Получение данных из запроса
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Отсутствуют данные в запросе'}), 400
        
        # Проверка обязательных полей
        required_fields = ['cpm', 'channel', 'date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Отсутствует обязательное поле: {field}'}), 400
        
        cpm = float(data['cpm'])
        channel = str(data['channel'])
        date_str = str(data['date'])
        
        # Валидация данных
        if cpm <= 0:
            return jsonify({'error': 'CPM должен быть положительным числом'}), 400
        
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Неверный формат даты. Используйте YYYY-MM-DD'}), 400
        
        # Предобработка
        X = preprocess_input(cpm, channel, date_str, label_encoders, feature_names)
        
        # Предсказание
        prediction = model.predict(X)[0]
        
        # Округление до целого (VIEWS не может быть отрицательным)
        predicted_views = max(0, int(round(prediction)))
        
        # Строгое соответствие требованиям: возвращаем только predicted_views
        return jsonify({
            'predicted_views': predicted_views
        })
        
    except ValueError as e:
        return jsonify({'error': f'Ошибка валидации данных: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Ошибка при предсказании: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint для массового прогнозирования
    
    Принимает JSON:
    {
        "predictions": [
            {"cpm": float, "channel": str, "date": str},
            ...
        ]
    }
    
    Возвращает:
    {
        "predictions": [
            {"cpm": float, "channel": str, "date": str, "predicted_views": int},
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Модель не загружена'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({'error': 'Отсутствует поле predictions'}), 400
        
        results = []
        for item in data['predictions']:
            try:
                cpm = float(item['cpm'])
                channel = str(item['channel'])
                date_str = str(item['date'])
                
                X = preprocess_input(cpm, channel, date_str, label_encoders, feature_names)
                prediction = model.predict(X)[0]
                predicted_views = max(0, int(round(prediction)))
                
                results.append({
                    'cpm': cpm,
                    'channel': channel,
                    'date': date_str,
                    'predicted_views': predicted_views
                })
            except Exception as e:
                results.append({
                    'cpm': item.get('cpm'),
                    'channel': item.get('channel'),
                    'date': item.get('date'),
                    'error': str(e)
                })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': f'Ошибка при обработке запроса: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ЗАПУСК API СЕРВЕРА")
    print("="*60)
    print("\nДоступные endpoints:")
    print("  GET  /health - проверка работоспособности")
    print("  POST /predict - прогнозирование одного объявления")
    print("  POST /predict_batch - массовое прогнозирование")
    print("\nПример запроса к /predict:")
    print('  {"cpm": 10.5, "channel": "example_channel", "date": "2024-10-02"}')
    print("\n" + "="*60)
    port = int(os.environ.get('PORT', '5000'))
    print(f"Сервер запущен на http://0.0.0.0:{port}")
    print("Используется Waitress (production-ready сервер)")
    print("="*60 + "\n")
    
    # Использование Waitress вместо встроенного Flask сервера
    serve(app, host='0.0.0.0', port=port)

