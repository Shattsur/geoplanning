# src\core\validator.py

"""Валидация данных и ограничений геопланирования."""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from math import asin, sqrt


def pairwise_haversine(points: np.ndarray, driving_factor: float = 1.4) -> np.ndarray:
    """
    Рассчитывает попарные расстояния между точками по формуле Хаверсина.
    
    Parameters:
    - points: массив формы (n, 2), где каждая строка [lat, lon]
    - driving_factor: коэффициент увеличения для учета реальных дорог
    
    Returns:
    - Матрицу расстояний формы (n, n) в километрах
    """
    n = len(points)
    if n <= 1:
        return np.zeros((n, n))
    
    R = 6371.0  # Радиус Земли в км
    
    # Конвертируем координаты в радианы
    lat_rad = np.radians(points[:, 0])
    lon_rad = np.radians(points[:, 1])
    
    # Создаем матрицы для векторизованных расчетов
    lat_i = lat_rad.reshape(-1, 1)
    lon_i = lon_rad.reshape(-1, 1)
    
    lat_j = lat_rad.reshape(1, -1)
    lon_j = lon_rad.reshape(1, -1)
    
    # Разницы координат
    dlat = lat_j - lat_i
    dlon = lon_j - lon_i
    
    # Формула Хаверсина (векторизованная)
    a = np.sin(dlat/2)**2 + np.cos(lat_i) * np.cos(lat_j) * np.sin(dlon/2)**2
    
    # Защита от численных ошибок (может быть чуть больше 1)
    a = np.where(a > 1.0, 1.0, a)
    
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c * driving_factor
    
    return distances


def validate_input_data(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """Базовая валидация входных данных."""
    errors = []
    warnings = []

    required = {'point_id', 'manager', 'lat', 'lon', 'n_visits'}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Отсутствуют обязательные колонки: {missing}")

    if df.empty:
        errors.append("DataFrame пустой")

    if not df['lat'].between(-90, 90).all():
        errors.append("Некорректные значения широты (должны быть в [-90, 90])")

    if not df['lon'].between(-180, 180).all():
        errors.append("Некорректные значения долготы (должны быть в [-180, 180])")

    if (df['n_visits'] < 1).any() or not df['n_visits'].dtype.kind in 'i':
        errors.append("n_visits должно быть целым числом >= 1")

    if not df['manager'].isin([0, 1, 2]).all():
        errors.append("manager должен принимать значения 0, 1 или 2")

    if df['point_id'].duplicated().any():
        dup_count = df['point_id'].duplicated().sum()
        warnings.append(f"Предупреждение: обнаружены дубликаты point_id ({dup_count} шт.). Рекомендуется переименовать или усреднить координаты.")

    return len(errors) == 0, errors, warnings


def check_cluster_constraints(
    labels: np.ndarray,
    points_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, List[Tuple[int, Any]]]:
    """
    Проверка ограничений кластеризации с учётом конфигурации.
    Добавлены:
    - min_points_per_cluster
    - min_visits_per_cluster (для дальних кластеров)
    - driving_factor из routing
    """
    clustering_cfg = config.get('clustering', {})
    routing_cfg = config.get('routing', {})

    max_visits = clustering_cfg.get('max_visits_per_cluster', 23)
    max_points = clustering_cfg.get('max_points_per_cluster', 18)
    min_points = clustering_cfg.get('min_points_per_cluster', 6)      # Новый: минимум точек в кластере
    min_visits = clustering_cfg.get('min_visits_per_cluster', 8)      # Новый: минимум визитов (для оправдания поездки)
    max_diameter = clustering_cfg.get('max_diameter_km', 50)
    driving_factor = routing_cfg.get('time', {}).get('driving_factor', 1.2)

    violations = {
        'overloaded_visits': [],      # слишком много визитов
        'too_many_points': [],        # слишком много точек
        'large_diameter': [],         # слишком большой диаметр
        'too_few_points': [],         # слишком мало точек (новое)
        'too_few_visits': [],         # слишком мало визитов (новое)
    }

    if len(labels) != len(points_df):
        raise ValueError(f"Длина labels ({len(labels)}) != длина DataFrame ({len(points_df)})")

    points_arr = points_df[['lat', 'lon']].values

    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        if cluster_id < 0:  # шум (DBSCAN) или изолированные — пропускаем или обрабатываем отдельно
            continue

        mask = labels == cluster_id
        cluster_df = points_df[mask]

        if cluster_df.empty:
            continue

        total_visits = int(cluster_df['n_visits'].sum())
        n_points = len(cluster_df)

        # Макс. ограничения
        if total_visits > max_visits:
            violations['overloaded_visits'].append((cluster_id, total_visits))

        if n_points > max_points:
            violations['too_many_points'].append((cluster_id, n_points))

        # Мин. ограничения (новые)
        if n_points < min_points:
            violations['too_few_points'].append((cluster_id, n_points))

        if total_visits < min_visits:
            violations['too_few_visits'].append((cluster_id, total_visits))

        # Диаметр с учётом реальных дорог
        if n_points >= 2:
            dist_matrix = pairwise_haversine(points_arr[mask], driving_factor=driving_factor)
            diameter_km = dist_matrix.max()
            if diameter_km > max_diameter:
                violations['large_diameter'].append((cluster_id, round(diameter_km, 2)))

    return violations


def check_schedule_intervals(
    routes: List[Dict],
    config: Dict[str, Any]
) -> List[str]:
    """
    Проверка интервалов для точек с n_visits = 2.
    Желательно: интервал ≈ полмесяца (≥14 дней).
    routes — список словарей с ключами: point_id, workday_number
    """
    scheduling_cfg = config.get('scheduling', {})
    repeat_gap_days = scheduling_cfg.get('repeat_gap_days', 15)
    violations = []

    point_visits = {}
    for route in routes:
        workday = route['workday_number']
        for point in route['points']:
            pid = point['point_id']
            if point['n_visits'] >= 2:  # только для повторных
                if pid not in point_visits:
                    point_visits[pid] = []
                point_visits[pid].append(workday)

    for pid, days in point_visits.items():
        if len(days) >= 2:
            days_sorted = sorted(days)
            gaps = [days_sorted[i+1] - days_sorted[i] for i in range(len(days_sorted)-1)]
            min_gap = min(gaps)
            if min_gap < repeat_gap_days - 3:  # допускаем небольшое отклонение
                violations.append(f"Точка {pid}: минимальный интервал {min_gap} дней (рекомендуется ≥{repeat_gap_days})")

    return violations


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Валидация структуры конфига."""
    errors = []

    required_sections = ['clustering', 'routing', 'scheduling', 'depot']
    for sec in required_sections:
        if sec not in config:
            errors.append(f"Отсутствует обязательная секция [{sec}] в конфиге")

    clustering = config.get('clustering', {})
    if clustering.get('max_visits_per_cluster', 0) <= 0:
        errors.append("clustering.max_visits_per_cluster должен быть > 0")
    if clustering.get('min_points_per_cluster', 0) < 1:
        errors.append("clustering.min_points_per_cluster должен быть >= 1")

    routing = config.get('routing', {})
    time_cfg = routing.get('time', {})
    if time_cfg.get('service_time_minutes', 0) <= 0:
        errors.append("routing.time.service_time_minutes должен быть > 0")
    if time_cfg.get('max_daily_hours', 0) <= 0:
        errors.append("routing.time.max_daily_hours должен быть > 0")

    return len(errors) == 0, errors


def check_violations_summary(violations: Dict[str, List]) -> str:
    """Краткая сводка нарушений."""
    counts = {k: len(v) for k, v in violations.items() if v}
    if not counts:
        return "Все кластеры соответствуют ограничениям ✅"
    return "Нарушения: " + ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in counts.items())