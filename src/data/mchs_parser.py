"""
Парсер официальных данных МЧС Кыргызстана.

Читает xlsx-файлы из data/open_data/ и извлекает:
- Зоны опасности паводков/селей по регионам
- Прорывоопасные озёра (категория I, II, III)
- Факторы активизации (ATM, PV, TEK)

Результат используется для калибровки пороговых значений модели
и добавления региональных признаков в датасет.
"""

from pathlib import Path

import pandas as pd


# Соответствие файлов и регионов
MCHS_FILES = {
    'talas': Path('data/open_data/- (2).xlsx'),
    'naryn': Path('data/open_data/- (3).xlsx'),
    'issyk_kul': Path('data/open_data/- (4).xlsx'),
    'jalal_abad': Path('data/open_data/- (5).xlsx'),
}

# Типы секций → ключевые слова в заголовке раздела
SECTION_PATTERNS = {
    'flood': ['сель', 'паводк'],
    'lake': ['прорывоопасных озер', 'прорывоопасных озёр'],
    'landslide': ['оползн'],
    'avalanche': ['лавин'],
    'waterlog': ['подтоплен'],
    'rockfall': ['камнепад', 'обвал'],
    'ice_jam': ['ледяных заторов'],
}

# Категории опасности → числовой балл
DANGER_SCORE = {'I': 3, 'II': 2, 'III': 1, '1': 3, '2': 2, '3': 1}


def _row_text(row: pd.Series) -> str:
    """Объединить непустые ячейки строки в одну строку."""
    return ' '.join(str(v) for v in row if str(v) not in ('nan', 'NaN', '')).strip()


def _detect_section(text: str) -> str | None:
    """Определить тип секции по строке заголовка. None если не секция."""
    if 'Прогноз возможн' not in text and 'Прогнозирован' not in text:
        return None
    text_lower = text.lower()
    for section, keywords in SECTION_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            return section
    return 'other'


def _is_table_header(row: pd.Series) -> bool:
    """Строка — заголовок таблицы данных."""
    vals = [str(v) for v in row if str(v) not in ('nan', 'NaN', '')]
    joined = ' '.join(vals)
    # Таблица паводков/оползней
    if any(k in joined for k in ('Айылный аймак', 'Айыльный аймак', 'Аыйлный аймак')):
        return True
    # Таблица озёр
    if 'Наименование озера' in joined or 'Категория опасности' in joined:
        return True
    return False


def _is_district_row(row: pd.Series) -> bool:
    """Строка — название района (короткая строка с 'район')."""
    vals = [str(v) for v in row if str(v) not in ('nan', 'NaN', '')]
    text = ' '.join(vals)
    return 'район' in text.lower() and len(vals) <= 4 and 'Прогноз' not in text


def _is_data_row(row: pd.Series) -> bool:
    """Строка — строка данных (первая непустая ячейка — число или номер вроде '7а')."""
    vals = [str(v) for v in row if str(v) not in ('nan', 'NaN', '')]
    if not vals:
        return False
    first = vals[0].strip()
    # Порядковый номер: цифра, цифра+буква, или "п" (шапка)
    if first in ('п', 'п.'):
        return False
    import re
    return bool(re.match(r'^\d+[а-яa-zА-ЯA-Z]*$', first))


def _extract_danger_category(vals: list[str]) -> str:
    """Найти категорию опасности I/II/III в списке значений строки."""
    for v in vals:
        v_clean = v.strip()
        if v_clean in ('I', 'II', 'III'):
            return v_clean
    return 'III'


def _extract_river(vals: list[str]) -> str:
    """Найти название реки/сая в строке данных."""
    river_keywords = ['р.', 'река', 'сай ', 'поток', 'канал', 'борт', 'лог ',
                      'оврагообраз', 'ручей', 'склон']
    for v in vals[1:]:
        v_lower = v.lower()
        if any(kw in v_lower for kw in river_keywords):
            return v.strip()
    return ''


def parse_region(region: str) -> dict:
    """
    Парсит xlsx-файл МЧС для региона.

    Возвращает словарь:
        {
            'flood_zones': int,       # число записей паводков/селей
            'lake_zones': int,        # число прорывоопасных озёр
            'lake_danger_i': int,     # озёра категории I (самые опасные)
            'lake_danger_ii': int,    # озёра категории II
            'lake_danger_iii': int,   # озёра категории III
            'rivers': list[str],      # уникальные реки/бассейны
        }
    """
    result = {
        'flood_zones': 0,
        'lake_zones': 0,
        'lake_danger_i': 0,
        'lake_danger_ii': 0,
        'lake_danger_iii': 0,
        'rivers': [],
    }

    if region not in MCHS_FILES:
        return result
    path = MCHS_FILES[region]
    if not path.exists():
        return result

    df = pd.read_excel(path, header=None)
    current_section = None
    in_data_table = False
    rivers_set: set[str] = set()

    for _, row in df.iterrows():
        text = _row_text(row)
        if not text:
            continue

        # 1. Определяем начало новой секции
        section = _detect_section(text)
        if section is not None:
            current_section = section
            in_data_table = False
            continue

        # 2. Заголовок таблицы данных
        if _is_table_header(row):
            in_data_table = True
            continue

        # 3. Сброс при новом районе
        if _is_district_row(row):
            # Не сбрасываем секцию — паводки могут идти по нескольким районам
            in_data_table = False
            continue

        # 4. Строки данных
        if not in_data_table or not _is_data_row(row):
            continue

        vals = [str(v) for v in row if str(v) not in ('nan', 'NaN', '')]

        if current_section == 'flood':
            result['flood_zones'] += 1
            river = _extract_river(vals)
            if river:
                rivers_set.add(river)

        elif current_section == 'lake':
            result['lake_zones'] += 1
            danger = _extract_danger_category(vals)
            if danger == 'I':
                result['lake_danger_i'] += 1
            elif danger == 'II':
                result['lake_danger_ii'] += 1
            else:
                result['lake_danger_iii'] += 1

            # Бассейн реки (озёра обычно в 5-й колонке)
            if len(vals) >= 5:
                rivers_set.add(vals[4][:30])

    result['rivers'] = sorted(rivers_set)
    return result


def get_region_stats() -> pd.DataFrame:
    """
    Агрегированная статистика по всем регионам из каталога МЧС.

    Колонки:
        region | flood_zones | lake_zones | lake_danger_i | lake_danger_ii |
        max_danger_score | flood_density_score | flood_density_norm
    """
    rows = []
    for region in MCHS_FILES:
        stats = parse_region(region)

        # Взвешенный балл опасности: озёра I весят в 3×, II в 2×, зоны паводков — 1×
        danger_score = (
            stats['lake_danger_i'] * 3
            + stats['lake_danger_ii'] * 2
            + stats['lake_danger_iii'] * 1
        )
        max_danger = (3 if stats['lake_danger_i'] > 0
                      else 2 if stats['lake_danger_ii'] > 0
                      else 1)
        density = float(danger_score * 2 + stats['flood_zones'])

        rows.append({
            'region': region,
            'flood_zones': stats['flood_zones'],
            'lake_zones': stats['lake_zones'],
            'lake_danger_i': stats['lake_danger_i'],
            'lake_danger_ii': stats['lake_danger_ii'],
            'max_danger_score': max_danger,
            'flood_density_score': density,
        })

    # Добавляем регионы без файлов МЧС с дефолтными значениями
    present = {r['region'] for r in rows}
    defaults = {
        'chui': (60, 3, 0, 2, 2, 66.0),
        'osh': (90, 5, 1, 2, 3, 105.0),
        'batken': (40, 2, 0, 1, 2, 44.0),
    }
    for region, (fz, lz, li, lii, md, ds) in defaults.items():
        if region not in present:
            rows.append({
                'region': region,
                'flood_zones': fz,
                'lake_zones': lz,
                'lake_danger_i': li,
                'lake_danger_ii': lii,
                'max_danger_score': md,
                'flood_density_score': ds,
            })

    df = pd.DataFrame(rows)
    max_density = df['flood_density_score'].max()
    df['flood_density_norm'] = (
        df['flood_density_score'] / max_density if max_density > 0 else 0.5
    )
    return df


def get_flood_threshold_multiplier(region: str, base_threshold: float) -> float:
    """
    Корректирует порог паводка на основе данных МЧС.
    Регионы с большим числом зон I категории имеют пониженный порог (до -30%).
    """
    stats = get_region_stats()
    row = stats[stats['region'] == region]
    if row.empty:
        return base_threshold
    density = float(row['flood_density_norm'].iloc[0])
    return base_threshold * (1.0 - 0.3 * density)


if __name__ == '__main__':
    print('=== Статистика регионов МЧС ===')
    df_stats = get_region_stats()
    print(df_stats.to_string(index=False))

    print('\n=== Детали по регионам ===')
    for region in MCHS_FILES:
        stats = parse_region(region)
        print(f'\n{region}:')
        print(f'  Зоны паводков/селей: {stats["flood_zones"]}')
        print(f'  Озёра: {stats["lake_zones"]} '
              f'(I: {stats["lake_danger_i"]}, '
              f'II: {stats["lake_danger_ii"]}, '
              f'III: {stats["lake_danger_iii"]})')
        if stats['rivers']:
            print(f'  Реки (первые 5): {stats["rivers"][:5]}')
