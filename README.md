# Компьютерное моделирование влияния осадков на возникновение паводков в районах Кыргызстана

## Описание

Современный программный комплекс для моделирования паводковых процессов в горных районах Кыргызстана с веб-интерфейсом, интеграцией реальных метеоданных и машинным обучением. Система позволяет анализировать влияние осадков на формирование поверхностного стока, прогнозировать возникновение паводков и оценивать риски.

## Основные возможности

### 🌐 Веб-интерфейс
- Интерактивный веб-интерфейс на Streamlit
- Интерактивные карты с OpenStreetMap
- Анимированные графики и визуализации
- Экспорт результатов в различных форматах
- Дашборд с ключевыми показателями

### 🌦️ Реальные метеоданные
- Интеграция с Open-Meteo API
- Почасовые данные о погоде
- Температура, осадки, влажность, скорость ветра
- Исторические данные и прогнозы
- Автоматическое кэширование запросов

### 🤖 Машинное обучение
- Прогнозирование уровня риска паводков
- Модели XGBoost и LightGBM
- Обучение на исторических данных
- Оценка важности признаков
- Экспорт обученных моделей

### 📊 Гидрологическое моделирование
- Расчёт поверхностного стока методом SCS Curve Number
- Моделирование таяния снега методом температурного индекса
- Построение гидрографов стока
- Выявление паводковых событий
- Сценарный анализ (нормальный, влажный, сухой, экстремальный)

### 📈 Визуализация
- Анимированные гидрографы с временной шкалой
- Тепловые карты календарных данных
- Сравнение сценариев
- Матрицы корреляций
- Карты рисков паводков
- Экспорт в PNG, SVG, HTML

## Структура проекта

```
diploma/
├── src/                              # Исходный код
│   ├── config.py                    # Конфигурация регионов
│   ├── data/                        # Модули работы с данными
│   │   ├── precipitation.py        # Данные об осадках
│   │   ├── terrain.py              # Анализ рельефа
│   │   ├── weather_api.py          # Интеграция Open-Meteo API
│   │   └── preprocessing.py        # Предобработка данных
│   ├── models/                      # Модели
│   │   ├── flood_model.py          # Гидрологическая модель
│   │   ├── scs_runoff.py           # Метод SCS Curve Number
│   │   ├── snowmelt.py             # Модель таяния снега
│   │   └── ml_predictor.py         # ML прогнозирование
│   ├── visualization/               # Визуализация
│   │   ├── plots.py                # Базовые графики
│   │   ├── maps.py                 # Карты рисков
│   │   ├── enhanced_charts.py      # Интерактивные графики
│   │   └── web_components.py       # Компоненты веб-интерфейса
│   └── utils/                       # Утилиты
│       └── helpers.py              # Вспомогательные функции
├── data/                            # Данные (загружаются автоматически)
├── notebooks/                       # Jupyter notebooks
│   └── demonstration.ipynb         # Демонстрация возможностей
├── tests/                           # Тесты
├── results/                         # Результаты моделирования
├── docs/                            # Документация
├── main.py                          # CLI скрипт
├── web_app.py                       # Веб-приложение Streamlit
├── run_web.sh                       # Скрипт запуска веб-интерфейса
├── requirements.txt                 # Зависимости Python
├── .gitignore                       # Git ignore файл
└── README.md                        # Этот файл
```

## Установка

```bash
# Клонирование репозитория
git clone <repository_url>
cd diploma

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### 🌐 Веб-интерфейс (рекомендуется)

Самый простой способ начать работу - использовать веб-интерфейс:

```bash
# Запуск веб-приложения
./run_web.sh

# Или напрямую
streamlit run web_app.py
```

Откройте браузер по адресу http://localhost:8501

**Возможности веб-интерфейса:**
- Выбор региона на интерактивной карте
- Выбор источника данных (реальные или синтетические)
- Настройка параметров моделирования
- Сравнение сценариев
- Обучение ML моделей
- Экспорт результатов
- Интерактивные графики и карты

### 💻 Командная строка

```bash
# Базовый запуск с синтетическими данными
python main.py --region chui --scenario normal --days 365

# Использование реальных данных из Open-Meteo API
python main.py --region issyk_kul --scenario normal --days 30 --use-real-data

# Доступные регионы: chui, issyk_kul, naryn, osh, jalal_abad, talas, batken
# Доступные сценарии: dry, normal, wet, extreme
```

### 🐍 Python API

#### Работа с синтетическими данными

```python
from src.models.flood_model import FloodModel
from src.utils.helpers import generate_sample_data

# Генерация тестовых данных
data = generate_sample_data(
    region='chui',
    days=365,
    scenario='normal',
    hourly=True  # Почасовые данные
)

# Создание и инициализация модели
model = FloodModel(region='chui')
model.initialize_state(data['timestamp'].iloc[0])

# Запуск моделирования
results = model.run_simulation(data)

# Выявление паводковых событий
events = model.detect_flood_events(results)
print(f"Обнаружено событий: {len(events)}")
```

#### Работа с реальными данными

```python
from src.data.weather_api import get_weather_data_for_model
from src.models.flood_model import FloodModel
from datetime import datetime, timedelta

# Получение реальных метеоданных
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

weather_data = get_weather_data_for_model(
    region='chui',
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d')
)

# Моделирование
model = FloodModel(region='chui')
results = model.run_simulation(weather_data)
```

#### Машинное обучение

```python
from src.models.ml_predictor import FloodMLPredictor
from src.utils.helpers import generate_sample_data

# Генерация обучающих данных
train_data = generate_sample_data(region='chui', days=730, hourly=True)

# Создание и обучение модели
ml_model = FloodMLPredictor(model_type='xgboost')
ml_model.train(train_data)

# Прогнозирование
predictions = ml_model.predict(train_data)
print(f"Точность модели: {ml_model.get_model_metrics()['accuracy']:.2%}")

# Сохранение модели
ml_model.save_model('models/flood_predictor.pkl')
```

### 📓 Jupyter Notebook

Откройте `notebooks/demonstration.ipynb` для интерактивного примера со всеми возможностями.

## Регионы Кыргызстана

| Регион | Код | Площадь (км²) | Ср. высота (м) |
|--------|-----|---------------|----------------|
| Чуйская область | chui | 20 200 | 1 500 |
| Иссык-Кульская область | issyk_kul | 43 100 | 2 500 |
| Нарынская область | naryn | 45 200 | 2 800 |
| Ошская область | osh | 29 200 | 2 200 |
| Джалал-Абадская область | jalal_abad | 33 700 | 2 000 |
| Таласская область | talas | 11 400 | 2 200 |
| Баткенская область | batken | 17 000 | 1 800 |

## Технологии и методы

### Гидрологическое моделирование

#### SCS Curve Number
Метод расчёта поверхностного стока на основе номера кривой стока:

```
Q = (P - Ia)² / (P - Ia + S)
S = 25400/CN - 254
Ia = 0.2 × S
```

Где:
- Q - поверхностный сток (мм)
- P - осадки (мм)
- S - потенциальное максимальное удержание (мм)
- CN - номер кривой стока (30-100)
- Ia - начальные потери

#### Модель таяния снега
Метод температурного индекса (degree-day):

```
M = DDF × (T - T_base)
```

Где:
- M - интенсивность таяния (мм/день)
- DDF - фактор степени-дня (2-6 мм/°C/день)
- T - температура воздуха (°C)
- T_base - базовая температура (обычно 0°C)

#### Единичный гидрограф SCS
Преобразование избыточных осадков в гидрограф стока для расчёта временного распределения стока.

### Машинное обучение

- **XGBoost**: Gradient Boosting для классификации уровня риска
- **LightGBM**: Быстрая альтернатива с меньшим потреблением памяти
- **Признаки**: осадки, температура, влажность, скорость ветра, накопленные осадки, характеристики водосбора
- **Метрики**: точность, precision, recall, F1-score, ROC-AUC

### Источники данных

- **Open-Meteo API**: Бесплатный API для получения метеоданных
  - Исторические данные с 1940 года
  - Почасовое разрешение
  - Глобальное покрытие
  - Без ограничений на запросы

- **Синтетические данные**: Генерация тестовых данных с учетом:
  - Сезонной вариативности
  - Высотной поясности
  - Экстремальных событий
  - Региональных особенностей Кыргызстана

### Технологический стек

**Backend:**
- Python 3.9+
- NumPy, Pandas - обработка данных
- Scikit-learn - ML инфраструктура
- XGBoost, LightGBM - модели машинного обучения
- Requests - HTTP клиент для API

**Visualization:**
- Matplotlib, Seaborn - статичные графики
- Plotly - интерактивные графики
- Folium - интерактивные карты

**Web Interface:**
- Streamlit - веб-фреймворк
- Streamlit-Folium - интеграция карт

## Результаты

Результаты моделирования автоматически сохраняются в директории `results/` и включают:

### Данные
- `simulation_results.csv` - временные ряды всех переменных
- `statistics.json` - статистические показатели
- `flood_events.json` - информация о паводковых событиях

### Визуализации
- Гидрографы стока (статические и анимированные)
- Календарные тепловые карты
- Карты риска паводков
- Графики сравнения сценариев
- Матрицы корреляций признаков

### Отчёты
- Текстовый отчёт с основными выводами
- HTML дашборды для веб-просмотра
- Экспортированные ML модели

## Требования

### Системные требования
- Python 3.9 или выше
- 4 GB RAM (рекомендуется 8 GB для ML моделей)
- 500 MB свободного места на диске
- Интернет-соединение (для загрузки реальных данных)

### Основные зависимости
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.2.0
- streamlit >= 1.28.0
- plotly >= 5.18.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0

Полный список зависимостей см. в `requirements.txt`

## Устранение неполадок

### Ошибки при установке зависимостей

Если возникают проблемы с установкой XGBoost или LightGBM на Windows:
```bash
# Используйте conda вместо pip
conda install -c conda-forge xgboost lightgbm
```

### Проблемы с Open-Meteo API

Если API недоступен:
```python
# Используйте синтетические данные
data = generate_sample_data(region='chui', days=365, hourly=True)
```

### Веб-интерфейс не запускается

```bash
# Убедитесь, что Streamlit установлен корректно
pip install --upgrade streamlit

# Проверьте версию Python
python --version  # Должна быть >= 3.9

# Запустите с явным указанием интерпретатора
python -m streamlit run web_app.py
```

## Примеры использования

### Быстрый старт

```python
# 1. Импорт модулей
from src.models.flood_model import FloodModel
from src.data.weather_api import get_weather_data_for_model
from datetime import datetime, timedelta

# 2. Получение данных за последний месяц
weather_data = get_weather_data_for_model(
    region='chui',
    start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
    end_date=datetime.now().strftime('%Y-%m-%d')
)

# 3. Моделирование
model = FloodModel(region='chui')
results = model.run_simulation(weather_data)

# 4. Анализ результатов
events = model.detect_flood_events(results)
print(f"Паводковых событий: {len(events)}")
print(f"Максимальный сток: {results['discharge'].max():.2f} м³/с")
```

### Сравнение сценариев

```python
from src.utils.helpers import generate_sample_data
from src.models.flood_model import FloodModel

scenarios = ['dry', 'normal', 'wet', 'extreme']
results = {}

for scenario in scenarios:
    data = generate_sample_data(region='chui', days=365, scenario=scenario)
    model = FloodModel(region='chui')
    results[scenario] = model.run_simulation(data)

# Сравнение максимального стока
for scenario, result in results.items():
    max_discharge = result['discharge'].max()
    print(f"{scenario}: {max_discharge:.2f} м³/с")
```

## Документация

- **Полная документация**: `docs/diploma_thesis.md`
- **Примеры использования**: `notebooks/demonstration.ipynb`
- **API документация**: встроенные docstrings в исходном коде

## Цитирование

Если вы используете этот проект в своей работе, пожалуйста, укажите:

```
Азамат Даниел. Компьютерное моделирование влияния осадков на возникновение паводков
в районах Кыргызстана. Дипломная работа. 2026.
```

## Благодарности

- **Open-Meteo** - за предоставление бесплатного API метеоданных
- **Streamlit** - за отличный фреймворк для создания веб-приложений
- Научному руководителю за поддержку и рекомендации

## Лицензия

MIT License - см. LICENSE файл для деталей

## Контакты

**Автор**: Daniel Azamat
**Email**: azamatdaniel0@gmail.com
**GitHub**: [azamatdaniel0](https://github.com/azamatdaniel0)

---

**Дипломная работа**
Кыргызский Государственный Технический Университет
2026
