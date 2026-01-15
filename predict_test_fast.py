"""
Быстрая версия обработки тестового датасета (оптимизированная)
"""
import pandas as pd
import numpy as np
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

def load_model_and_encoders():
    """Загрузка модели и энкодеров"""
    print("Загрузка модели и энкодеров...")
    
    model = joblib.load('model_cb.pkl')
    print("Модель загружена: CB")
    
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    print(f"Энкодеры и признаки загружены ({len(feature_names)} признаков)")
    return model, label_encoders, feature_names

def preprocess_fast(df, label_encoders, feature_names):
    """Быстрая предобработка с оптимизацией"""
    df = df.copy()
    
    # Очистка названий колонок
    df.columns = df.columns.str.strip()
    
    # Обработка дат (векторизованные операции)
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
    
    # Кодирование каналов
    if 'channel' in label_encoders:
        le_channel = label_encoders['channel']
        known_channels = set(le_channel.classes_)
        df['channel_encoded'] = df['CHANNEL_NAME'].apply(
            lambda x: le_channel.transform([x])[0] if x in known_channels else -1
        ).astype('int32')
    else:
        df['channel_encoded'] = 0
    
    # Статистики по каналам (оптимизированный merge)
    if 'channel_stats' in label_encoders:
        channel_stats = label_encoders['channel_stats']
        global_avg_views = label_encoders.get('global_avg_views', 100)
        global_median_views = label_encoders.get('global_median_views', 50)
        global_std_views = label_encoders.get('global_std_views', 50)
        avg_cpm = float(df['CPM'].mean())
        
        # Merge по колонке (быстрее)
        df = df.merge(channel_stats, on='CHANNEL_NAME', how='left', sort=False)
        
        # Заполнение пропусков (векторизованные операции)
        df['channel_avg_views'] = df['channel_avg_views'].fillna(global_avg_views).astype('float32')
        df['channel_median_views'] = df['channel_median_views'].fillna(global_median_views).astype('float32')
        df['channel_std_views'] = df['channel_std_views'].fillna(global_std_views).astype('float32')
        df['channel_count'] = df['channel_count'].fillna(1).astype('float32')
        df['channel_avg_cpm'] = df['channel_avg_cpm'].fillna(avg_cpm).astype('float32')
        
        if 'channel_avg_clicks' in channel_stats.columns:
            df['channel_avg_clicks'] = df['channel_avg_clicks'].fillna(0).astype('float32')
        else:
            df['channel_avg_clicks'] = 0.0
        
        if 'channel_avg_actions' in channel_stats.columns:
            df['channel_avg_actions'] = df['channel_avg_actions'].fillna(0).astype('float32')
        else:
            df['channel_avg_actions'] = 0.0
    else:
        df['channel_avg_views'] = label_encoders.get('global_avg_views', 100)
        df['channel_median_views'] = label_encoders.get('global_median_views', 50)
        df['channel_std_views'] = label_encoders.get('global_std_views', 50)
        df['channel_count'] = 1
        df['channel_avg_cpm'] = df['CPM'].mean()
        df['channel_avg_clicks'] = 0
        df['channel_avg_actions'] = 0
    
    # CTR и CVR (векторизованные операции)
    df['ctr'] = (df['channel_avg_clicks'] / (df['channel_avg_views'] + 1)).fillna(0).astype('float32')
    df['cvr'] = (df['channel_avg_actions'] / (df['channel_avg_views'] + 1)).fillna(0).astype('float32')
    
    # Создание всех признаков
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Выбор признаков
    X = df[feature_names].copy()
    X = X.fillna(0).astype('float32')
    
    return X

def main():
    print("="*60)
    print("БЫСТРАЯ ОБРАБОТКА ТЕСТОВОГО ДАТАСЕТА")
    print("="*60)
    
    # Загрузка модели
    model, label_encoders, feature_names = load_model_and_encoders()
    
    # Загрузка данных
    print("\nЗагрузка TestDataset.csv...")
    start_load = time.time()
    test_df = pd.read_csv('TestDataset.csv', dtype={'CPM': 'float32'}, engine='c')
    load_time = time.time() - start_load
    print(f"Загрузка завершена за {load_time:.1f} секунд")
    print(f"Размер: {test_df.shape}")
    
    # Предобработка
    print("\nПредобработка данных...")
    start_preprocess = time.time()
    X = preprocess_fast(test_df, label_encoders, feature_names)
    preprocess_time = time.time() - start_preprocess
    print(f"Предобработка завершена за {preprocess_time:.1f} секунд")
    
    # Предсказания
    print("\nВыполнение предсказаний...")
    start_predict = time.time()
    batch_size = 100000  # Большой батч для ускорения
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X.iloc[i:end_idx]
        predictions[i:end_idx] = model.predict(batch_X)
        print(f"Обработано: {end_idx:,}/{n_samples:,} ({100*end_idx/n_samples:.1f}%)")
    
    predict_time = time.time() - start_predict
    print(f"Предсказания завершены за {predict_time:.1f} секунд")
    
    # Сохранение
    print("\nСохранение результатов...")
    test_df['VIEWS'] = np.maximum(0, np.round(predictions).astype(int))
    test_df.to_csv('TestDataset_predictions.csv', index=False)
    
    print("\n" + "="*60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"Время: {load_time + preprocess_time + predict_time:.1f} секунд")
    print("Файл: TestDataset_predictions.csv")
    print("="*60)

if __name__ == "__main__":
    main()



