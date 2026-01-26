# Получение данных о паводках для обучения модели

## Статус базы данных DFO (Dartmouth Flood Observatory)

**Обновление: Январь 2026**

Веб-сайт DFO недавно был переработан, и база данных FloodArchive.csv временно недоступна для скачивания.

**Официальный сайт**: https://floodobservatory.colorado.edu/

**Контакт для запроса данных**: kettner@colorado.edu

---

## Вариант 1: Запросить данные у DFO (Рекомендуется)

Напишите email на kettner@colorado.edu с просьбой предоставить доступ к FloodArchive.csv или аналогичным данным.

**Пример письма**:
```
Subject: Request for Flood Archive Data for Research

Dear DFO Team,

I am conducting research on flood prediction in Kyrgyzstan using machine learning
methods. For my thesis work, I need historical flood event data from your database.

Could you please provide access to the FloodArchive.csv file or direct me to where
I can download it? I specifically need data on flood events in Kyrgyzstan and
Central Asia.

The data will be used solely for academic research purposes.

Thank you for your assistance.

Best regards,
[Your Name]
[Institution: Kyrgyz State Technical University]
[Email]
```

---

## Вариант 2: Альтернативные источники данных о паводках

### 2.1 EM-DAT (Emergency Events Database)

**Сайт**: https://www.emdat.be/

**Как получить**:
1. Зарегистрируйтесь на сайте (бесплатно для академических целей)
2. Поиск: Country = Kyrgyzstan, Disaster Type = Flood
3. Скачайте результаты в CSV

**Преимущества**:
- Бесплатный доступ для исследователей
- Включает крупные бедствия
- Данные с 1900 года

**Недостатки**:
- Только крупные события (мелкие паводки могут отсутствовать)

### 2.2 Министерство чрезвычайных ситуаций Кыргызстана

**Сайт**: https://mes.kg/

**Как получить**:
1. Обратитесь в пресс-службу МЧС
2. Запросите исторические данные о паводках
3. Укажите, что данные нужны для дипломной работы

**Преимущества**:
- Наиболее точные локальные данные
- Включает все события на территории Кыргызстана
- Официальные записи

**Недостатки**:
- Требуется время на получение ответа
- Возможно, потребуется официальный запрос от университета

### 2.3 Кыргызгидромет

**Организация**: Гидрометеорологическая служба (часть МЧС)

**Контакт**: Через МЧС или напрямую в региональных отделениях

**Преимущества**:
- Данные о расходах рек (измерения на постах)
- Исторические записи осадков
- Региональная детализация

---

## Вариант 3: Работа без исторических данных (временное решение)

Система может работать и без базы данных паводков, используя **пороговый метод**:

```python
from src.models.ml_predictor import generate_training_data_real

# Без flood_events будет использован пороговый метод
training_data = generate_training_data_real(
    region='chui',
    start_date='2018-01-01',
    end_date='2023-12-31',
    flood_events=None  # Пороговый метод: discharge > 100 м³/с = паводок
)
```

**Недостатки**:
- Менее точное определение паводков
- Модель учится на искусственных метках, а не реальных событиях
- Пороговое значение (100 м³/с) может не соответствовать реальным условиям

**Когда использовать**:
- Для тестирования системы
- Для демонстрации функциональности
- Пока ожидаете получения реальных данных о паводках

---

## Вариант 4: Создание собственной базы данных

Если получение данных затруднено, можно создать базу данных вручную:

### 4.1 Источники для поиска информации о паводках

1. **Новостные архивы**:
   - Кыргызские новостные сайты (24.kg, Kloop.kg, Азаттык)
   - Поиск по ключевым словам: "паводок", "наводнение", "селевой поток"

2. **Научные публикации**:
   - Google Scholar: "floods in Kyrgyzstan"
   - Российские научные журналы по гидрологии

3. **Международные отчеты**:
   - ReliefWeb (https://reliefweb.int)
   - GDACS (Global Disaster Alert and Coordination System)

### 4.2 Формат базы данных

Создайте CSV файл `data/kyrgyzstan_floods_manual.csv`:

```csv
start_date,end_date,Country,severity,Dead,Displaced,MainCause
2023-05-15,2023-05-18,Kyrgyzstan - Chui,high,3,150,Heavy Rain
2022-07-08,2022-07-12,Kyrgyzstan - Osh,critical,7,1200,Monsoon
2021-08-19,2021-08-22,Kyrgyzstan - Jalal-Abad,moderate,1,430,Heavy Rain
```

**Колонки**:
- `start_date`: Дата начала паводка (YYYY-MM-DD)
- `end_date`: Дата окончания (YYYY-MM-DD)
- `Country`: Страна и регион
- `severity`: Уровень серьезности (moderate/high/critical)
- `Dead`: Количество погибших
- `Displaced`: Количество пострадавших
- `MainCause`: Причина (Heavy Rain, Snowmelt, Monsoon, etc.)

**Использование**:
```python
from src.data.flood_database import load_kyrgyzstan_floods

floods = load_kyrgyzstan_floods('data/kyrgyzstan_floods_manual.csv')
```

---

## Текущий статус

- ✅ Система для работы с реальными данными готова
- ✅ Код для загрузки и обработки паводков реализован
- ⏳ База данных DFO временно недоступна
- 🔄 Возможна работа с пороговым методом или альтернативными источниками

---

## Рекомендуемый план действий

1. **Сейчас**: Протестируйте систему с пороговым методом (без базы данных)
   ```bash
   python scripts/train_real_data.py --region chui
   ```

2. **Параллельно**: Отправьте запрос на kettner@colorado.edu

3. **Альтернатива**: Зарегистрируйтесь на EM-DAT и скачайте данные

4. **Для диплома**: Обратитесь в МЧС Кыргызстана за официальными данными

5. **Когда получите данные**: Перезапустите обучение с реальными метками
   ```bash
   python scripts/train_real_data.py --region chui --dfo-path data/FloodArchive.csv
   ```

---

## Документация для модуля flood_database

Подробнее о функциях работы с данными паводков см. в файле:
[src/data/flood_database.py](../src/data/flood_database.py)

Функции:
- `load_kyrgyzstan_floods()` - загрузка базы данных DFO
- `merge_flood_labels()` - присвоение меток паводков
- `filter_floods_by_region()` - фильтрация по регионам
- `get_flood_statistics()` - статистика событий

---

**Обновлено**: 26 января 2026
