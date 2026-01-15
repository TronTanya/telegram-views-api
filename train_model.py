"""
Обучение модели машинного обучения для прогнозирования VIEWS
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_train=True, label_encoders=None):
    """
    Предобработка данных
    """
    df = df.copy()
    
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
    
    # Дополнительные временные признаки
    df['is_month_start'] = df['DATE'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['DATE'].dt.is_month_end.astype(int)
    
    # Кодирование каналов
    if label_encoders is None:
        label_encoders = {}
    
    if 'CHANNEL_NAME' in df.columns:
        if 'channel' not in label_encoders:
            le_channel = LabelEncoder()
            df['channel_encoded'] = le_channel.fit_transform(df['CHANNEL_NAME'])
            label_encoders['channel'] = le_channel
        else:
            le_channel = label_encoders['channel']
            # Для неизвестных каналов используем -1
            known_channels = set(le_channel.classes_)
            df['channel_encoded'] = df['CHANNEL_NAME'].apply(
                lambda x: le_channel.transform([x])[0] if x in known_channels else -1
            )
    
    # Статистики по каналам будут созданы позже (после обработки CLICKS/ACTIONS)
    # Обработка тестовой выборки
    if not is_train:
        # Для тестовой выборки используем сохраненные статистики
        if 'channel_stats' in label_encoders:
            channel_stats = label_encoders['channel_stats']
            df = df.merge(channel_stats, on='CHANNEL_NAME', how='left')
            # Заполнение пропусков средними значениями
            df['channel_avg_views'] = df['channel_avg_views'].fillna(label_encoders.get('global_avg_views', 100))
            df['channel_median_views'] = df['channel_median_views'].fillna(label_encoders.get('global_median_views', 50))
            df['channel_std_views'] = df['channel_std_views'].fillna(label_encoders.get('global_std_views', 50))
            df['channel_count'] = df['channel_count'].fillna(1)
            df['channel_avg_cpm'] = df['channel_avg_cpm'].fillna(df['CPM'].mean())
        else:
            # Если статистик нет, создаем пустые признаки
            df['channel_avg_views'] = 100
            df['channel_median_views'] = 50
            df['channel_std_views'] = 50
            df['channel_count'] = 1
            df['channel_avg_cpm'] = df['CPM'].mean()
    
    # Дополнительные признаки из расширенного датасета (если доступны)
    if 'CLICKS' in df.columns:
        # CTR (Click-Through Rate) - отношение кликов к просмотрам
        # Используем историческое среднее для канала, если VIEWS нет
        if 'VIEWS' in df.columns:
            df['ctr'] = (df['CLICKS'] / (df['VIEWS'] + 1)).fillna(0)
        else:
            df['ctr'] = 0  # Для тестовой выборки без VIEWS
    
    if 'ACTIONS' in df.columns:
        # CVR (Conversion Rate) - отношение действий к просмотрам
        if 'VIEWS' in df.columns:
            df['cvr'] = (df['ACTIONS'] / (df['VIEWS'] + 1)).fillna(0)
        else:
            df['cvr'] = 0
    
    # Статистики по каналам для CLICKS и ACTIONS (если доступны)
    if is_train and 'VIEWS' in df.columns:
        agg_dict = {
            'VIEWS': ['mean', 'median', 'std', 'count'],
            'CPM': 'mean'
        }
        if 'CLICKS' in df.columns:
            agg_dict['CLICKS'] = 'mean'
        if 'ACTIONS' in df.columns:
            agg_dict['ACTIONS'] = 'mean'
        
        channel_stats = df.groupby('CHANNEL_NAME').agg(agg_dict).reset_index()
        
        # Формирование названий колонок
        new_cols = ['CHANNEL_NAME', 'channel_avg_views', 'channel_median_views', 
                   'channel_std_views', 'channel_count', 'channel_avg_cpm']
        if 'CLICKS' in df.columns:
            new_cols.append('channel_avg_clicks')
        if 'ACTIONS' in df.columns:
            new_cols.append('channel_avg_actions')
        
        channel_stats.columns = new_cols
        df = df.merge(channel_stats, on='CHANNEL_NAME', how='left')
        
        # Заполнение пропусков для новых каналов
        df['channel_avg_views'] = df['channel_avg_views'].fillna(df['VIEWS'].mean())
        df['channel_median_views'] = df['channel_median_views'].fillna(df['VIEWS'].median())
        df['channel_std_views'] = df['channel_std_views'].fillna(df['VIEWS'].std())
        df['channel_count'] = df['channel_count'].fillna(1)
        df['channel_avg_cpm'] = df['channel_avg_cpm'].fillna(df['CPM'].mean())
        if 'channel_avg_clicks' in df.columns:
            df['channel_avg_clicks'] = df['channel_avg_clicks'].fillna(0)
        if 'channel_avg_actions' in df.columns:
            df['channel_avg_actions'] = df['channel_avg_actions'].fillna(0)
    
    # Выбор признаков для модели
    feature_cols = [
        'CPM',
        'channel_encoded',
        'year', 'month', 'day', 'day_of_week', 'day_of_year', 
        'week_of_year', 'is_weekend', 'quarter',
        'is_month_start', 'is_month_end',
        'channel_avg_views', 'channel_median_views', 
        'channel_std_views', 'channel_count', 'channel_avg_cpm'
    ]
    
    # Добавляем дополнительные признаки, если они есть
    if 'ctr' in df.columns:
        feature_cols.append('ctr')
    if 'cvr' in df.columns:
        feature_cols.append('cvr')
    if 'channel_avg_clicks' in df.columns:
        feature_cols.append('channel_avg_clicks')
    if 'channel_avg_actions' in df.columns:
        feature_cols.append('channel_avg_actions')
    
    # Проверка наличия признаков
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    
    # Заполнение пропусков
    X = X.fillna(X.median())
    
    if is_train and 'VIEWS' in df.columns:
        y = df['VIEWS'].values
        return X, y, label_encoders
    else:
        return X, label_encoders

def train_models(X_train, y_train, X_val, y_val):
    """
    Обучение нескольких моделей и выбор лучшей
    """
    models = {}
    results = {}
    
    # XGBoost
    print("Обучение XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    models['xgb'] = xgb_model
    y_pred_xgb = xgb_model.predict(X_val)
    results['xgb'] = {
        'mae': mean_absolute_error(y_val, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_xgb)),
        'r2': r2_score(y_val, y_pred_xgb)
    }
    print(f"XGBoost - MAE: {results['xgb']['mae']:.2f}, RMSE: {results['xgb']['rmse']:.2f}, R2: {results['xgb']['r2']:.4f}")
    
    # LightGBM
    print("Обучение LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbose=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    models['lgb'] = lgb_model
    y_pred_lgb = lgb_model.predict(X_val)
    results['lgb'] = {
        'mae': mean_absolute_error(y_val, y_pred_lgb),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_lgb)),
        'r2': r2_score(y_val, y_pred_lgb)
    }
    print(f"LightGBM - MAE: {results['lgb']['mae']:.2f}, RMSE: {results['lgb']['rmse']:.2f}, R2: {results['lgb']['r2']:.4f}")
    
    # CatBoost
    print("Обучение CatBoost...")
    cb_model = cb.CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models['cb'] = cb_model
    y_pred_cb = cb_model.predict(X_val)
    results['cb'] = {
        'mae': mean_absolute_error(y_val, y_pred_cb),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred_cb)),
        'r2': r2_score(y_val, y_pred_cb)
    }
    print(f"CatBoost - MAE: {results['cb']['mae']:.2f}, RMSE: {results['cb']['rmse']:.2f}, R2: {results['cb']['r2']:.4f}")
    
    # Выбор лучшей модели по MAE
    best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = models[best_model_name]
    
    print(f"\nЛучшая модель: {best_model_name.upper()}")
    print(f"MAE: {results[best_model_name]['mae']:.2f}")
    print(f"RMSE: {results[best_model_name]['rmse']:.2f}")
    print(f"R2: {results[best_model_name]['r2']:.4f}")
    
    return best_model, best_model_name, results

def load_training_data():
    """
    Загрузка обучающих данных из AllData.csv
    """
    import os
    
    if os.path.exists('AllData.csv'):
        print("Используется AllData.csv (обучающий датасет)")
        return pd.read_csv('AllData.csv')
    else:
        raise FileNotFoundError(
            "Не найден файл с обучающими данными. "
            "Поместите AllData.csv в текущую директорию."
        )

def main():
    """
    Основная функция обучения
    """
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    train_df = load_training_data()
    print(f"Размер обучающей выборки: {train_df.shape}")
    
    # Очистка названий колонок от пробелов
    train_df.columns = train_df.columns.str.strip()
    print(f"\nДоступные колонки: {list(train_df.columns)}")
    
    # Проверка обязательных колонок
    required_cols = ['CPM', 'CHANNEL_NAME', 'DATE', 'VIEWS']
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
    
    # Показываем дополнительную информацию о данных
    if 'CLICKS' in train_df.columns:
        print(f"Дополнительные данные: CLICKS, ACTIONS доступны")
    if 'AD_ID' in train_df.columns:
        print(f"Уникальных объявлений: {train_df['AD_ID'].nunique() if 'AD_ID' in train_df.columns else 'N/A'}")
    
    # Предобработка
    print("\nПредобработка данных...")
    X, y, label_encoders = preprocess_data(train_df, is_train=True)
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Признаки: {list(X.columns)}")
    
    # Сохранение статистик каналов для тестовой выборки
    if 'channel_stats' not in label_encoders:
        agg_dict = {
            'VIEWS': ['mean', 'median', 'std', 'count'],
            'CPM': 'mean'
        }
        if 'CLICKS' in train_df.columns:
            agg_dict['CLICKS'] = 'mean'
        if 'ACTIONS' in train_df.columns:
            agg_dict['ACTIONS'] = 'mean'
        
        channel_stats = train_df.groupby('CHANNEL_NAME').agg(agg_dict).reset_index()
        
        # Формирование названий колонок
        new_cols = ['CHANNEL_NAME', 'channel_avg_views', 'channel_median_views', 
                   'channel_std_views', 'channel_count', 'channel_avg_cpm']
        if 'CLICKS' in train_df.columns:
            new_cols.append('channel_avg_clicks')
        if 'ACTIONS' in train_df.columns:
            new_cols.append('channel_avg_actions')
        
        channel_stats.columns = new_cols
        label_encoders['channel_stats'] = channel_stats
        label_encoders['global_avg_views'] = y.mean()
        label_encoders['global_median_views'] = np.median(y)
        label_encoders['global_std_views'] = y.std()
    
    # Разделение на train/val
    print("\nРазделение на train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Обучение моделей
    print("\nОбучение моделей...")
    best_model, best_model_name, results = train_models(X_train, y_train, X_val, y_val)
    
    # Сохранение модели и энкодеров
    print("\nСохранение модели...")
    joblib.dump(best_model, f'model_{best_model_name}.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')
    
    print(f"\nМодель сохранена: model_{best_model_name}.pkl")
    print(f"Энкодеры сохранены: label_encoders.pkl")
    print(f"Имена признаков сохранены: feature_names.pkl")
    
    # Сохранение метрик
    with open('model_metrics.txt', 'w', encoding='utf-8') as f:
        f.write("МЕТРИКИ МОДЕЛИ\n")
        f.write("="*60 + "\n")
        f.write(f"Лучшая модель: {best_model_name.upper()}\n")
        f.write(f"MAE: {results[best_model_name]['mae']:.2f}\n")
        f.write(f"RMSE: {results[best_model_name]['rmse']:.2f}\n")
        f.write(f"R2: {results[best_model_name]['r2']:.4f}\n")
        f.write("\nВсе модели:\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name.upper()}:\n")
            f.write(f"  MAE: {metrics['mae']:.2f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  R2: {metrics['r2']:.4f}\n")
    
    print("\nМетрики сохранены: model_metrics.txt")
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)

if __name__ == "__main__":
    main()

