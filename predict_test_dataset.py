"""
Скрипт для заполнения TestDataset.csv предсказаниями модели
"""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта tqdm для прогресс-бара
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Для отображения прогресса установите tqdm: pip install tqdm")

def load_model_and_encoders():
    """Загрузка модели и энкодеров"""
    print("Загрузка модели и энкодеров...")
    
    # Загрузка модели
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
    
    # Загрузка энкодеров и признаков
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    print(f"Энкодеры и признаки загружены")
    print(f"Количество признаков: {len(feature_names)}")
    
    return model, model_name, label_encoders, feature_names

def preprocess_test_data(df, label_encoders, feature_names):
    """
    Предобработка тестовых данных (аналогично train_model.py)
    """
    df = df.copy()
    
    # Очистка названий колонок от пробелов
    df.columns = df.columns.str.strip()
    
    # Обработка дат (оптимизированные типы данных)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['year'] = df['DATE'].dt.year.astype('int16')
    df['month'] = df['DATE'].dt.month.astype('int8')
    df['day'] = df['DATE'].dt.day.astype('int8')
    df['day_of_week'] = df['DATE'].dt.dayofweek.astype('int8')
    df['day_of_year'] = df['DATE'].dt.dayofyear.astype('int16')
    df['week_of_year'] = df['DATE'].dt.isocalendar().week.astype('int8')
    df['is_weekend'] = (df['DATE'].dt.dayofweek >= 5).astype('int8')
    df['quarter'] = df['DATE'].dt.quarter.astype('int8')
    df['is_month_start'] = df['DATE'].dt.is_month_start.astype('int8')
    df['is_month_end'] = df['DATE'].dt.is_month_end.astype('int8')
    
    # Кодирование каналов (оптимизированное - использование словаря вместо apply)
    if 'channel' in label_encoders:
        le_channel = label_encoders['channel']
        # Создаем словарь для быстрого поиска (правильный порядок: idx, channel)
        channel_to_code = {channel: idx for idx, channel in enumerate(le_channel.classes_)}
        # Используем map вместо apply - намного быстрее
        df['channel_encoded'] = df['CHANNEL_NAME'].map(channel_to_code).fillna(-1).astype('int32')
    else:
        df['channel_encoded'] = 0
    
    # Статистики по каналам (максимально оптимизированный merge)
    if 'channel_stats' in label_encoders:
        channel_stats = label_encoders['channel_stats']
        global_avg_views = label_encoders.get('global_avg_views', 100)
        global_median_views = label_encoders.get('global_median_views', 50)
        global_std_views = label_encoders.get('global_std_views', 50)
        avg_cpm = float(df['CPM'].mean())
        
        # Используем merge по колонке (оптимизировано для скорости)
        df = df.merge(channel_stats, on='CHANNEL_NAME', how='left', sort=False)
        
        # Заполнение пропусков для неизвестных каналов (векторизованная операция с оптимизированными типами)
        df['channel_avg_views'] = df['channel_avg_views'].fillna(global_avg_views).astype('float32')
        df['channel_median_views'] = df['channel_median_views'].fillna(global_median_views).astype('float32')
        df['channel_std_views'] = df['channel_std_views'].fillna(global_std_views).astype('float32')
        df['channel_count'] = df['channel_count'].fillna(1).astype('float32')
        df['channel_avg_cpm'] = df['channel_avg_cpm'].fillna(avg_cpm).astype('float32')
        
        # Дополнительные статистики (если доступны)
        if 'channel_avg_clicks' in channel_stats.columns:
            df['channel_avg_clicks'] = df['channel_avg_clicks'].fillna(0).astype('float32')
        else:
            df['channel_avg_clicks'] = 0.0
        
        if 'channel_avg_actions' in channel_stats.columns:
            df['channel_avg_actions'] = df['channel_avg_actions'].fillna(0).astype('float32')
        else:
            df['channel_avg_actions'] = 0.0
    else:
        # Если статистик нет
        df['channel_avg_views'] = label_encoders.get('global_avg_views', 100)
        df['channel_median_views'] = label_encoders.get('global_median_views', 50)
        df['channel_std_views'] = label_encoders.get('global_std_views', 50)
        df['channel_count'] = 1
        df['channel_avg_cpm'] = df['CPM'].mean()
        df['channel_avg_clicks'] = 0
        df['channel_avg_actions'] = 0
    
    # Дополнительные признаки (ctr, cvr) - оптимизированные типы
    df['ctr'] = (df['channel_avg_clicks'] / (df['channel_avg_views'] + 1)).fillna(0).astype('float32')
    df['cvr'] = (df['channel_avg_actions'] / (df['channel_avg_views'] + 1)).fillna(0).astype('float32')
    
    # Убеждаемся, что все необходимые признаки созданы
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Выбор признаков в правильном порядке
    X = df[feature_names].copy()
    
    # Заполнение пропусков и преобразование в float32 (оптимизированно)
    X = X.fillna(0).astype('float32')
    
    return X

def main():
    """
    Основная функция для обработки тестового датасета
    """
    print("="*60)
    print("ОБРАБОТКА ТЕСТОВОГО ДАТАСЕТА")
    print("="*60)
    
    # Загрузка модели
    model, model_name, label_encoders, feature_names = load_model_and_encoders()
    
    # Загрузка тестового датасета (оптимизированная)
    print("\nЗагрузка TestDataset.csv...")
    print("Обрабатывается 318,722+ записей, пожалуйста, подождите...")
    print("Это может занять 30-60 секунд...")
    
    # Оптимизация загрузки больших CSV
    import time
    start_load = time.time()
    
    # Используем оптимизированные параметры для быстрой загрузки
    test_df = pd.read_csv(
        'TestDataset.csv',
        dtype={
            'CPM': 'float32',
            'VIEWS': 'float32'  # Если есть пустые значения
        },
        low_memory=False,
        engine='c'  # Используем C парсер (быстрее)
    )
    
    load_time = time.time() - start_load
    print(f"Загрузка завершена за {load_time:.1f} секунд")
    print(f"Размер тестового датасета: {test_df.shape}")
    print(f"Колонки: {list(test_df.columns)}")
    
    # Проверка структуры
    required_cols = ['CPM', 'CHANNEL_NAME', 'DATE']
    missing_cols = [col for col in required_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
    
    # Предобработка данных
    print("\nПредобработка данных...")
    print("Это может занять 30-60 секунд для большого датасета...")
    start_preprocess = time.time()
    
    if HAS_TQDM:
        print("Обработка данных...")
    X = preprocess_test_data(test_df, label_encoders, feature_names)
    
    preprocess_time = time.time() - start_preprocess
    print(f"Предобработка завершена за {preprocess_time:.1f} секунд")
    print(f"Количество признаков: {X.shape[1]}")
    
    # Предсказание
    print("\nВыполнение предсказаний...")
    print("Это может занять некоторое время...")
    print(f"Всего записей для обработки: {len(X):,}")
    
    # Для больших датасетов используем батчевую обработку
    batch_size = 50000  # Увеличиваем размер батча для ускорения
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=np.float32)
    
    if HAS_TQDM:
        iterator = tqdm(range(0, n_samples, batch_size), desc="Предсказания", unit="записей")
    else:
        iterator = range(0, n_samples, batch_size)
        print(f"Обработка {n_samples:,} записей батчами по {batch_size:,}...")
        start_time = pd.Timestamp.now()
    
    for batch_num, i in enumerate(iterator, 1):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X.iloc[i:end_idx]
        predictions[i:end_idx] = model.predict(batch_X)
        
        if not HAS_TQDM:
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            rate = end_idx / elapsed if elapsed > 0 else 0
            remaining = (n_samples - end_idx) / rate if rate > 0 else 0
            print(f"Батч {batch_num}: {end_idx:,}/{n_samples:,} ({100*end_idx/n_samples:.1f}%) | "
                  f"Скорость: {rate:.0f} записей/сек | Осталось: ~{remaining:.0f} сек")
    
    # Округление до целых и гарантия неотрицательных значений
    predictions = np.maximum(0, np.round(predictions).astype(int))
    
    # Заполнение столбца VIEWS
    test_df['VIEWS'] = predictions
    
    # Сохранение результата (оптимизированное)
    output_file = 'TestDataset_predictions.csv'
    print(f"\nСохранение результатов в {output_file}...")
    print("Это может занять некоторое время для больших файлов...")
    # Используем оптимизированные параметры для ускорения
    test_df.to_csv(output_file, index=False, chunksize=10000)
    
    # Статистика предсказаний
    print("\n" + "="*60)
    print("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
    print("="*60)
    print(f"Всего записей: {len(predictions)}")
    print(f"Минимальное VIEWS: {predictions.min()}")
    print(f"Максимальное VIEWS: {predictions.max()}")
    print(f"Среднее VIEWS: {predictions.mean():.2f}")
    print(f"Медианное VIEWS: {np.median(predictions):.2f}")
    print(f"Сумма всех VIEWS: {predictions.sum():,}")
    
    # Топ-10 каналов по среднему предсказанному VIEWS
    print("\nТоп-10 каналов по среднему предсказанному VIEWS:")
    channel_stats = test_df.groupby('CHANNEL_NAME').agg({
        'VIEWS': ['mean', 'count']
    }).round(2)
    channel_stats.columns = ['avg_views', 'count']
    channel_stats = channel_stats.sort_values('avg_views', ascending=False).head(10)
    print(channel_stats)
    
    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()

