"""
Скрипт для тестирования API
"""
import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Тест проверки работоспособности"""
    print("Тест: GET /health")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✓ Успешно\n")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}\n")
        return False

def test_predict():
    """Тест прогнозирования одного объявления"""
    print("Тест: POST /predict")
    try:
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
        print(f"Статус: {response.status_code}")
        print(f"Запрос: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✓ Успешно\n")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}\n")
        return False

def test_predict_batch():
    """Тест массового прогнозирования"""
    print("Тест: POST /predict_batch")
    try:
        data = {
            "predictions": [
                {"cpm": 10.5, "channel": "tanya_in_france", "date": "2024-10-02"},
                {"cpm": 15.0, "channel": "relocator_cc", "date": "2024-10-03"},
                {"cpm": 5.0, "channel": "teleportazia", "date": "2024-10-04"}
            ]
        }
        response = requests.post(
            f"{API_URL}/predict_batch",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Статус: {response.status_code}")
        print(f"Запрос: {json.dumps(data, indent=2, ensure_ascii=False)}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        print("✓ Успешно\n")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}\n")
        return False

def test_validation():
    """Тест валидации данных"""
    print("Тест: Валидация данных")
    
    # Тест с отрицательным CPM
    print("\n1. Тест с отрицательным CPM:")
    try:
        data = {"cpm": -10, "channel": "test", "date": "2024-01-01"}
        response = requests.post(f"{API_URL}/predict", json=data)
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест с неверным форматом даты
    print("\n2. Тест с неверным форматом даты:")
    try:
        data = {"cpm": 10, "channel": "test", "date": "01-01-2024"}
        response = requests.post(f"{API_URL}/predict", json=data)
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест с отсутствующим полем
    print("\n3. Тест с отсутствующим полем:")
    try:
        data = {"cpm": 10, "channel": "test"}
        response = requests.post(f"{API_URL}/predict", json=data)
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    print()

def main():
    """Основная функция тестирования"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ API")
    print("="*60)
    print(f"API URL: {API_URL}\n")
    
    # Проверка доступности API
    if not test_health():
        print("API недоступен. Убедитесь, что сервер запущен (python api.py)")
        return
    
    # Тесты
    test_predict()
    test_predict_batch()
    test_validation()
    
    print("="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)

if __name__ == "__main__":
    main()



