# src/strategies/exclusion/distance_based.py

"""Финальная стратегия отсеивания: по расстоянию + по общей нагрузке (для single-депо)."""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

from src.core.interfaces import ExclusionStrategy, Point, Depot


def haversine_vectorized(lat1_array, lon1_array, lat2, lon2):
    """
    Векторизованная функция расстояния Хаверсина.
    Рассчитывает расстояния между массивами точек и одной точкой.
    
    Returns: расстояния в километрах
    """
    R = 6371.0  # Радиус Земли в км
    
    # Конвертируем в радианы
    lat1_rad = np.radians(lat1_array)
    lon1_rad = np.radians(lon1_array)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Разницы
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Формула Хаверсина
    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


class DistanceBasedExclusion(ExclusionStrategy):
    """
    Умное отсеивание точек:
    - Сначала по расстоянию от депо
    - Затем по общей нагрузке (чтобы влезть в план на 22 дня)
    - Приоритет: снижать повторные визиты (2→1), потом отсеивать одиночные
    """

    def __init__(self, use_real_distances: bool = False):
        self.use_real_distances = use_real_distances

    def exclude(
        self,
        points: List[Point],
        depot: Depot,
        config: Dict[str, Any]
    ) -> Tuple[List[Point], List[Point]]:
        exclusion_cfg = config.get("exclusion", {})
        if not exclusion_cfg.get("enabled", True):
            print("Отсев отключён в конфиге")
            return points, []

        max_dist_km = exclusion_cfg.get("max_dist_from_depot_km", 80.0)
        max_exclusion_percent = exclusion_cfg.get("max_exclusion_percent", 45) / 100.0
        priority_reduce = exclusion_cfg.get("priority_reduce", True)

        if not points:
            return [], []

        total_visits_initial = sum(p.n_visits for p in points)
        max_exclude_visits = int(total_visits_initial * max_exclusion_percent)

        # Целевая нагрузка на месяц (из конфига или расчёт)
        working_days = config["project"]["working_days"].get("december", 22)
        max_visits_per_day = config.get("scheduling", {}).get("daily_distribution", {}).get("max_visits_per_day", 25)
        target_total_visits = working_days * max_visits_per_day

        # Если визитов больше, чем нужно — включаем отсев по нагрузке
        if total_visits_initial <= target_total_visits:
            print(f"Нагрузка {total_visits_initial} визитов ≤ цель {target_total_visits} — отсев только по расстоянию")
            max_exclude_visits = 0  # не отсеиваем по нагрузке
        else:
            needed_reduce = total_visits_initial - target_total_visits
            max_exclude_visits = max(max_exclude_visits, needed_reduce)
            print(f"Нагрузка {total_visits_initial} > цель {target_total_visits} → нужно снизить на {needed_reduce} визитов")

        # Расстояния от депо
        lats = np.array([p.lat for p in points])
        lons = np.array([p.lon for p in points])
        distances = haversine_vectorized(lats, lons, np.array([depot.lat]), np.array([depot.lon])).flatten()

        df = pd.DataFrame({
            'point': points,
            'distance_km': distances,
            'n_visits': [p.n_visits for p in points]
        })

        # Сортировка: дальние + низкий приоритет (n_visits=1) первыми на отсев
        df = df.sort_values(by=['distance_km', 'n_visits'], ascending=[False, True]).reset_index(drop=True)

        kept_points: List[Point] = []
        excluded_points: List[Point] = []
        reduced_count = 0
        excluded_count = 0
        current_excluded = 0

        for _, row in df.iterrows():
            point = row['point']
            dist = row['distance_km']
            visits = point.n_visits

            if current_excluded >= max_exclude_visits:
                kept_points.append(point)
                continue

            remaining = max_exclude_visits - current_excluded

            # 1. Отсев по расстоянию (жёсткий)
            if dist > max_dist_km:
                if remaining >= visits:
                    excluded_points.append(point)
                    excluded_count += visits
                    current_excluded += visits
                    print(f"  Отсеяна по расстоянию: {point.point_id} ({visits} виз.), {dist:.1f} км")
                else:
                    kept_points.append(point)
                continue

            # 2. Снижение повторных визитов (если нужно по нагрузке)
            if priority_reduce and visits > 1 and remaining >= 1:
                new_point = Point(
                    point_id=point.point_id,
                    manager=point.manager,
                    lat=point.lat,
                    lon=point.lon,
                    n_visits=1
                )
                kept_points.append(new_point)
                reduced_count += (visits - 1)
                current_excluded += (visits - 1)
                print(f"  Снижен приоритет: {point.point_id} ({visits}→1), {dist:.1f} км")
                continue

            # 3. Отсев одиночных визитов (если всё ещё нужно снижать нагрузку)
            if visits == 1 and remaining >= 1:
                excluded_points.append(point)
                excluded_count += 1
                current_excluded += 1
                print(f"  Отсеяна одиночная: {point.point_id}, {dist:.1f} км")
                continue

            # Иначе оставляем
            kept_points.append(point)

        total_excluded = reduced_count + excluded_count
        final_visits = total_visits_initial - total_excluded
        coverage = final_visits / total_visits_initial if total_visits_initial > 0 else 0

        print(f"\nОТСЕВ ЗАВЕРШЁН:")
        print(f"  • Снижено визитов (2→1): {reduced_count}")
        print(f"  • Полностью отсеяно: {len(excluded_points)} точек ({excluded_count} визитов)")
        print(f"  • Всего исключено: {total_excluded} визитов ({total_excluded / total_visits_initial:.1%})")
        print(f"  • Осталось: {final_visits} визитов ({coverage:.1%} от исходных)")

        return kept_points, excluded_points