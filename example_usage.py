"""
Примеры использования API для прогнозирования VIEWS
"""
import requests
import json

API_URL = "http://localhost:5000"

def example_single_prediction():
    """Пример прогнозирования одного объявления"""
    print("Пример 1: Прогнозирование одного объявления")
    print("-" * 60)
    
    data = {
        "cpm": 10.5,
        "channel": "tanya_in_france",
        "date": "2024-10-02"
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Входные данные:")
        print(f"  CPM: {data['cpm']}")
        print(f"  Канал: {data['channel']}")
        print(f"  Дата: {data['date']}")
        print(f"\nПрогноз:")
        print(f"  Предсказанный VIEWS: {result['predicted_views']}")
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    print("\n")

def example_batch_prediction():
    """Пример массового прогнозирования"""
    print("Пример 2: Массовое прогнозирование")
    print("-" * 60)
    
    data = {
        "predictions": [
            {"cpm": 10.5, "channel": "tanya_in_france", "date": "2024-10-02"},
            {"cpm": 15.0, "channel": "relocator_cc", "date": "2024-10-03"},
            {"cpm": 5.0, "channel": "teleportazia", "date": "2024-10-04"},
            {"cpm": 20.0, "channel": "pitkvch_news", "date": "2024-10-05"}
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict_batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        results = response.json()['predictions']
        print(f"Обработано объявлений: {len(results)}\n")
        
        for i, result in enumerate(results, 1):
            print(f"Объявление {i}:")
            print(f"  CPM: {result['cpm']}")
            print(f"  Канал: {result['channel']}")
            print(f"  Дата: {result['date']}")
            print(f"  Предсказанный VIEWS: {result['predicted_views']}")
            print()
    else:
        print(f"Ошибка: {response.status_code}")
        print(response.json())
    
    print("\n")

def example_comparison():
    """Пример сравнения прогнозов для разных CPM"""
    print("Пример 3: Сравнение прогнозов для разных CPM")
    print("-" * 60)
    
    channel = "tanya_in_france"
    date = "2024-10-02"
    cpm_values = [5.0, 10.0, 15.0, 20.0, 25.0]
    
    print(f"Канал: {channel}")
    print(f"Дата: {date}\n")
    print(f"{'CPM':<10} {'Предсказанный VIEWS':<25}")
    print("-" * 35)
    
    for cpm in cpm_values:
        data = {
            "cpm": cpm,
            "channel": channel,
            "date": date
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{cpm:<10} {result['predicted_views']:<25}")
        else:
            print(f"{cpm:<10} Ошибка")
    
    print("\n")

def example_different_channels():
    """Пример сравнения прогнозов для разных каналов"""
    print("Пример 4: Сравнение прогнозов для разных каналов")
    print("-" * 60)
    
    cpm = 10.0
    date = "2024-10-02"
    channels = ["tanya_in_france", "relocator_cc", "teleportazia", "pitkvch_news"]
    
    print(f"CPM: {cpm}")
    print(f"Дата: {date}\n")
    print(f"{'Канал':<25} {'Предсказанный VIEWS':<25}")
    print("-" * 50)
    
    for channel in channels:
        data = {
            "cpm": cpm,
            "channel": channel,
            "date": date
        }
        
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"{channel:<25} {result['predicted_views']:<25}")
        else:
            print(f"{channel:<25} Ошибка")
    
    print("\n")

if __name__ == "__main__":
    print("=" * 60)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ API")
    print("=" * 60)
    print(f"\nУбедитесь, что API сервер запущен на {API_URL}\n")
    
    try:
        # Проверка доступности API
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("API недоступен. Запустите сервер: python api.py")
            exit(1)
        
        # Примеры
        example_single_prediction()
        example_batch_prediction()
        example_comparison()
        example_different_channels()
        
        print("=" * 60)
        print("ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"Не удалось подключиться к API на {API_URL}")
        print("Убедитесь, что сервер запущен: python api.py")
    except Exception as e:
        print(f"Ошибка: {e}")



