"""
Exploratory Data Analysis (EDA) для данных рекламных кампаний Telegram
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')
sns.set_palette("husl")

def load_data(train_path=None, test_path=None):
    """
    Загрузка обучающих и тестовых данных
    Обучающие данные: AllData.csv
    Тестовые данные: TestDataset.csv
    """
    import os
    
    print("Загрузка данных...")
    
    # Определение пути к обучающим данным
    if train_path is None:
        if os.path.exists('AllData.csv'):
            train_path = 'AllData.csv'
            print("Используется AllData.csv (обучающий датасет)")
        else:
            raise FileNotFoundError(
                "Не найден файл с обучающими данными. "
                "Поместите AllData.csv в текущую директорию."
            )
    
    train_df = pd.read_csv(train_path)
    # Очистка названий колонок от пробелов
    train_df.columns = train_df.columns.str.strip()
    
    # Определение пути к тестовым данным
    if test_path is None:
        if os.path.exists('TestDataset.csv'):
            test_path = 'TestDataset.csv'
            print("Используется TestDataset.csv (тестовый датасет)")
        else:
            test_path = None
    
    # Загрузка тестовых данных (если файл существует)
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        # Очистка названий колонок от пробелов
        test_df.columns = test_df.columns.str.strip()
        print(f"Обучающая выборка: {train_df.shape}")
        print(f"Тестовая выборка: {test_df.shape}")
    else:
        test_df = None
        print(f"Обучающая выборка: {train_df.shape}")
        print("Тестовый файл не найден: TestDataset.csv")
    
    return train_df, test_df

def basic_info(df, name="Dataset"):
    """Базовая информация о датасете"""
    print(f"\n{'='*60}")
    print(f"БАЗОВАЯ ИНФОРМАЦИЯ: {name}")
    print(f"{'='*60}")
    print(f"Размер: {df.shape}")
    print(f"\nТипы данных:")
    print(df.dtypes)
    print(f"\nПервые строки:")
    print(df.head())
    print(f"\nОписательная статистика:")
    print(df.describe())
    print(f"\nПропущенные значения:")
    print(df.isnull().sum())
    print(f"\nДубликаты: {df.duplicated().sum()}")

def preprocess_dates(df):
    """Предобработка дат"""
    df = df.copy()
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['year'] = df['DATE'].dt.year
        df['month'] = df['DATE'].dt.month
        df['day'] = df['DATE'].dt.day
        df['day_of_week'] = df['DATE'].dt.dayofweek
        df['day_of_year'] = df['DATE'].dt.dayofyear
        df['week_of_year'] = df['DATE'].dt.isocalendar().week
        df['is_weekend'] = (df['DATE'].dt.dayofweek >= 5).astype(int)
    return df

def analyze_target_variable(df):
    """Анализ целевой переменной VIEWS"""
    if 'VIEWS' not in df.columns or df['VIEWS'].isna().all():
        print("\nЦелевая переменная отсутствует или пуста")
        return
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (VIEWS)")
    print(f"{'='*60}")
    
    views = df['VIEWS'].dropna()
    print(f"Среднее: {views.mean():.2f}")
    print(f"Медиана: {views.median():.2f}")
    print(f"Стандартное отклонение: {views.std():.2f}")
    print(f"Минимум: {views.min()}")
    print(f"Максимум: {views.max()}")
    print(f"Квантили:")
    print(views.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    
    # Распределение
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(views, bins=50, edgecolor='black')
    plt.title('Распределение VIEWS')
    plt.xlabel('VIEWS')
    plt.ylabel('Частота')
    
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(views), bins=50, edgecolor='black')
    plt.title('Распределение log(VIEWS + 1)')
    plt.xlabel('log(VIEWS + 1)')
    plt.ylabel('Частота')
    
    plt.subplot(1, 3, 3)
    plt.boxplot(views)
    plt.title('Boxplot VIEWS')
    plt.ylabel('VIEWS')
    
    plt.tight_layout()
    plt.savefig('views_distribution.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: views_distribution.png")

def analyze_cpm(df):
    """Анализ CPM"""
    print(f"\n{'='*60}")
    print("АНАЛИЗ CPM")
    print(f"{'='*60}")
    
    # Проверка наличия колонки CPM
    if 'CPM' not in df.columns:
        print("Колонка 'CPM' не найдена в датасете.")
        print(f"Доступные колонки: {list(df.columns)}")
        return
    
    cpm = df['CPM'].dropna()
    print(f"Среднее: {cpm.mean():.2f}")
    print(f"Медиана: {cpm.median():.2f}")
    print(f"Минимум: {cpm.min()}")
    print(f"Максимум: {cpm.max()}")
    
    # Взаимосвязь CPM и VIEWS
    if 'VIEWS' in df.columns and not df['VIEWS'].isna().all():
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(df['CPM'], df['VIEWS'], alpha=0.3, s=10)
        plt.xlabel('CPM')
        plt.ylabel('VIEWS')
        plt.title('CPM vs VIEWS')
        
        plt.subplot(1, 2, 2)
        # Логарифмическая шкала
        plt.scatter(df['CPM'], np.log1p(df['VIEWS']), alpha=0.3, s=10)
        plt.xlabel('CPM')
        plt.ylabel('log(VIEWS + 1)')
        plt.title('CPM vs log(VIEWS + 1)')
        
        plt.tight_layout()
        plt.savefig('cpm_views_relationship.png', dpi=150, bbox_inches='tight')
        print("График сохранен: cpm_views_relationship.png")

def analyze_channels(df):
    """Анализ каналов"""
    print(f"\n{'='*60}")
    print("АНАЛИЗ КАНАЛОВ")
    print(f"{'='*60}")
    
    print(f"Уникальных каналов: {df['CHANNEL_NAME'].nunique()}")
    print(f"\nТоп-10 каналов по количеству объявлений:")
    print(df['CHANNEL_NAME'].value_counts().head(10))
    
    if 'VIEWS' in df.columns and not df['VIEWS'].isna().all():
        channel_stats = df.groupby('CHANNEL_NAME').agg({
            'VIEWS': ['mean', 'median', 'count'],
            'CPM': 'mean'
        }).round(2)
        channel_stats.columns = ['avg_views', 'median_views', 'count', 'avg_cpm']
        channel_stats = channel_stats.sort_values('avg_views', ascending=False)
        
        print(f"\nТоп-10 каналов по среднему VIEWS:")
        print(channel_stats.head(10))
        
        # Визуализация
        plt.figure(figsize=(15, 6))
        top_channels = channel_stats.head(20)
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(top_channels)), top_channels['avg_views'])
        plt.yticks(range(len(top_channels)), top_channels.index)
        plt.xlabel('Средний VIEWS')
        plt.title('Топ-20 каналов по среднему VIEWS')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        plt.scatter(top_channels['avg_cpm'], top_channels['avg_views'], s=100, alpha=0.6)
        plt.xlabel('Средний CPM')
        plt.ylabel('Средний VIEWS')
        plt.title('CPM vs VIEWS по каналам')
        
        plt.tight_layout()
        plt.savefig('channels_analysis.png', dpi=150, bbox_inches='tight')
        print("График сохранен: channels_analysis.png")

def analyze_temporal(df):
    """Анализ временных паттернов"""
    if 'DATE' not in df.columns:
        return
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ВРЕМЕННЫХ ПАТТЕРНОВ")
    print(f"{'='*60}")
    
    df = preprocess_dates(df)
    
    if 'VIEWS' in df.columns and not df['VIEWS'].isna().all():
        # По дням недели
        weekday_stats = df.groupby('day_of_week')['VIEWS'].mean()
        print("\nСредний VIEWS по дням недели:")
        days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        for i, day in enumerate(days):
            if i in weekday_stats.index:
                print(f"{day}: {weekday_stats[i]:.2f}")
        
        # По месяцам
        month_stats = df.groupby('month')['VIEWS'].mean()
        print("\nСредний VIEWS по месяцам:")
        print(month_stats)
        
        # Визуализация
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        weekday_stats.plot(kind='bar')
        plt.xlabel('День недели')
        plt.ylabel('Средний VIEWS')
        plt.title('VIEWS по дням недели')
        plt.xticks(range(7), days, rotation=0)
        
        plt.subplot(2, 2, 2)
        month_stats.plot(kind='bar')
        plt.xlabel('Месяц')
        plt.ylabel('Средний VIEWS')
        plt.title('VIEWS по месяцам')
        
        plt.subplot(2, 2, 3)
        df.groupby('DATE')['VIEWS'].mean().plot()
        plt.xlabel('Дата')
        plt.ylabel('Средний VIEWS')
        plt.title('VIEWS во времени')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        df.groupby('DATE')['CPM'].mean().plot()
        plt.xlabel('Дата')
        plt.ylabel('Средний CPM')
        plt.title('CPM во времени')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=150, bbox_inches='tight')
        print("График сохранен: temporal_analysis.png")

def correlation_analysis(df):
    """Анализ корреляций"""
    if 'VIEWS' not in df.columns or df['VIEWS'].isna().all():
        return
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ КОРРЕЛЯЦИЙ")
    print(f"{'='*60}")
    
    df = preprocess_dates(df)
    numeric_cols = ['CPM', 'VIEWS', 'CLICKS', 'ACTIONS', 'year', 'month', 
                    'day', 'day_of_week', 'day_of_year', 'week_of_year', 'is_weekend']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    corr_matrix = df[numeric_cols].corr()
    
    print("\nКорреляция с VIEWS:")
    if 'VIEWS' in corr_matrix.columns:
        view_corr = corr_matrix['VIEWS'].sort_values(ascending=False)
        print(view_corr)
    
    # Визуализация
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("График сохранен: correlation_matrix.png")

def main():
    """Основная функция EDA"""
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Прогнозирование охвата рекламных объявлений в Telegram")
    print("="*60)
    
    # Загрузка данных
    train_df, test_df = load_data()
    
    # Базовая информация
    basic_info(train_df, "Обучающая выборка")
    if test_df is not None:
        basic_info(test_df, "Тестовая выборка")
    
    # Предобработка дат
    train_df = preprocess_dates(train_df)
    if test_df is not None:
        test_df = preprocess_dates(test_df)
    
    # Анализ целевой переменной
    analyze_target_variable(train_df)
    
    # Анализ CPM
    analyze_cpm(train_df)
    
    # Анализ каналов
    analyze_channels(train_df)
    
    # Временной анализ
    analyze_temporal(train_df)
    
    # Корреляционный анализ
    correlation_analysis(train_df)
    
    print("\n" + "="*60)
    print("EDA ЗАВЕРШЕН")
    print("="*60)

if __name__ == "__main__":
    main()

