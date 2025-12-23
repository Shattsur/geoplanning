# src/main.py

"""Основной пайплайн геопланировщика: от загрузка данных до результатов."""

import os
import yaml
from pathlib import Path
import sys
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
import requests  # Для проверки OSRM

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_and_prepare_data
from src.strategies.exclusion.distance_based import DistanceBasedExclusion
from src.strategies.clustering.kmeans_optimized import KMeansCapacitated
from src.strategies.routing.ortools_vrp import ORToolsVRPStrategy
from src.strategies.scheduling.load_balancing import LoadBalancingScheduling
from src.visualization.map_visualizer import save_folium_map, generate_yandex_links
from src.visualization.report_generator import generate_pdf_report
from src.core.interfaces import Depot


def get_working_days(config: Dict[str, Any]) -> List[datetime]:
    month_name = config["scheduling"].get("month", "december").lower()
    year = datetime.now().year
    months = {"december": 12, "january": 1}
    month_num = months.get(month_name, 12)
    start = datetime(year, month_num, 1)
    end = (start + pd.DateOffset(months=1) - pd.Timedelta(days=1))

    ru_holidays = [
        datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3), datetime(2025, 1, 4),
        datetime(2025, 1, 5), datetime(2025, 1, 6), datetime(2025, 1, 7), datetime(2025, 1, 8),
        datetime(2025, 2, 23), datetime(2025, 3, 8), datetime(2025, 5, 1), datetime(2025, 5, 9),
        datetime(2025, 6, 12), datetime(2025, 11, 4)
    ]

    dates = pd.date_range(start, end)
    working_dates = [d for d in dates if d.weekday() < 5 and d not in ru_holidays]
    num_days = config["project"]["working_days"].get(month_name, config["project"]["working_days"]["default"])
    working_dates = working_dates[:num_days]

    print(f"Рабочие дни для {month_name} 2025: {len(working_dates)} дней")
    return working_dates


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Конфигурация загружена из {config_path}")
    return config


def create_depots(config: Dict[str, Any], manager_ids: List[int]) -> Dict[int, Depot]:
    depot_cfg = config["depot"]
    strategy = depot_cfg["strategy"]

    if strategy == "custom":
        depots = {}
        for mgr_str, depot_data in depot_cfg["custom"].items():
            mgr_id = int(mgr_str)
            depots[mgr_id] = Depot(
                lat=depot_data["lat"],
                lon=depot_data["lon"],
                name=depot_data.get("name", f"Депо менеджера {mgr_id}")
            )
        return depots

    elif strategy == "single":
        single_cfg = depot_cfg["single_location"]
        single_depot = Depot(
            lat=single_cfg["lat"],
            lon=single_cfg["lon"],
            name=single_cfg.get("name", "Общее депо")
        )
        return {mid: single_depot for mid in manager_ids}

    elif strategy == "centroid":
        return None

    else:
        raise ValueError(f"Неизвестная стратегия депо: {strategy}")

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
    
    # Защита от численных ошибок
    a = np.where(a > 1.0, 1.0, a)
    
    c = 2 * np.arcsin(np.sqrt(a))
    distances = R * c * driving_factor
    
    return distances

def run_pipeline(
    data_path: str | None = None,
    config_path: str = "configs/default_config.yaml",
    verbose: bool = True,
    config_override: Dict[str, Any] = None
) -> Dict[str, Any]:
    config = load_config(config_path)

    if config_override:
        for section, params in config_override.items():
            config.setdefault(section, {}).update(params)

    # === Проверка OSRM в начале ===
    print(f"\n{'='*20} ПРОВЕРКА OSRM {'='*20}")
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        if response.status_code == 200:
            print("OSRM сервер доступен и работает ✓")
        else:
            print(f"OSRM ответил кодом {response.status_code} — возможно, данные не загружены")
    except Exception as e:
        print(f"OSRM НЕДОСТУПЕН: {e}")
        print("   → Запустите start_osrm.bat в отдельном окне перед запуском пайплайна")
        print("   → Если сервер не запускается — дистанции будут 0 км")
    print(f"{'='*60}\n")

    # 1. Загрузка данных
    all_points, depots_from_loader = load_and_prepare_data(config, data_path)
    manager_ids = sorted({p.manager for p in all_points})

    # 2. Депо
    if config["depot"]["strategy"] == "centroid":
        depots_by_manager = depots_from_loader
    else:
        depots_by_manager = create_depots(config, manager_ids)

    if not depots_by_manager:
        raise ValueError("Не удалось создать депо — проверь конфиг")

    print(f"Используется стратегия депо: {config['depot']['strategy']}")
    for mid, depot in depots_by_manager.items():
        print(f"  Менеджер {mid}: {depot.name} ({depot.lat:.6f}, {depot.lon:.6f})")

    results = {
        "managers": {},
        "excluded_points": [],
        "total_visits_planned": 0,
        "total_distance_km": 0.0,
        "map_paths": {},
        "yandex_links_path": None
    }

    base_dir = Path(config["output"]["directories"]["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = base_dir / "maps"
    reports_dir = base_dir / "reports"
    maps_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    working_days = get_working_days(config)

    for manager_id in manager_ids:
        depot = depots_by_manager[manager_id]
        manager_points = [p for p in all_points if p.manager == manager_id]

        print(f"\n{'='*60}")
        print(f"МЕНЕДЖЕР {manager_id} | Исходно: {len(manager_points)} точек, "
              f"{sum(p.n_visits for p in manager_points)} визитов")
        print(f"{'='*60}")

        # Отсеивание
        excluder = DistanceBasedExclusion(use_real_distances=False)
        kept_points, excluded_points = excluder.exclude(manager_points, depot, config)

        results["excluded_points"].extend(excluded_points)
        print(f"После отсеивания: {len(kept_points)} точек (отсеяно: {len(excluded_points)})")

        if not kept_points:
            print(f"ВНИМАНИЕ: Все точки менеджера {manager_id} отсеяны!")
            results["managers"][manager_id] = {"routes": [], "clusters": []}
            continue

        # Кластеризация
        print(f"\n--- Кластеризация ---")
        clusterer = KMeansCapacitated(verbose=verbose)
        clusterer.fit_predict(kept_points, config)
        clusters = clusterer.get_clusters()
        print(f"Сформировано кластеров: {len(clusters)}")

        for i, cluster in enumerate(clusters):
            pts_count = len(cluster.points)
            visits = sum(p.n_visits for p in cluster.points)
            coords = np.array([[p.lat, p.lon] for p in cluster.points])
            diameter_km = pairwise_haversine(coords).max() if pts_count > 1 else 0.0
            print(f"  Кластер {i:2d}: {pts_count:3d} точек | {visits:3d} визитов | диаметр ≈ {diameter_km:5.1f} км")

        # Маршрутизация
        print(f"\n--- Маршрутизация (до назначения дней) ---")
        router = ORToolsVRPStrategy(verbose=verbose)
        routes_without_days = []
        for cluster in clusters:
            route = router.optimize(cluster, depot, config)
            routes_without_days.append(route)

            print(f"  Маршрут кластера {route.route_id.split('_')[-1]}: "
                  f"{len(route.points)} точек | {route.total_visits} визитов | "
                  f"{route.total_distance_km:.1f} км | {route.estimated_driving_hours:.1f} ч в пути")

        # Планирование по дням
        print(f"\n--- Назначение рабочих дней ---")
        scheduler = LoadBalancingScheduling()
        routes = scheduler.schedule(routes_without_days, working_days, config)

        day_stats = {}
        for route in routes:
            if route.workday_numbers:
                day_str = route.workday_numbers[0]
                day_stats.setdefault(day_str, []).append(route)

        for day in sorted(day_stats.keys()):
            day_routes = day_stats[day]
            total_visits = sum(r.total_visits for r in day_routes)
            print(f"  {day}: {len(day_routes)} маршрут(ов) | {total_visits} визитов всего")

        # === ИСПРАВЛЕНИЕ: правильный подсчёт использованных дней ===
        # Извлекаем уникальные даты из назначенных маршрутов
        assigned_days = {r.workday_numbers[0] for r in routes if r.workday_numbers}
        used_days = len(assigned_days)

        # Сохраняем результаты по менеджеру
        visits_planned = sum(r.total_visits for r in routes)
        distance_km = sum(r.total_distance_km for r in routes)

        results["managers"][manager_id] = {
            "depot": depot,
            "clusters": clusters,
            "routes": routes,
            "visits_planned": visits_planned,
            "distance_km": distance_km,
            "used_days": used_days  # ← Сохраняем правильное значение
        }

        results["total_visits_planned"] += visits_planned
        results["total_distance_km"] += distance_km

        # Визуализация — один раз
        map_path = maps_dir / f"manager_{manager_id}_routes.html"
        save_folium_map(
            routes=routes,
            depot=depot,
            output_path=map_path,
            clusters=clusters,
            excluded_points=excluded_points,
            manager_id=manager_id
        )
        results["map_paths"][manager_id] = map_path
        print(f"\nКарта сохранена: {map_path}")

    # Yandex-ссылки
    if results["managers"]:
        all_routes = [r for m in results["managers"].values() for r in m["routes"]]
        yandex_links_path = base_dir / "yandex_routes_links.xlsx"
        generate_yandex_links(all_routes, depots_by_manager, yandex_links_path)
        results["yandex_links_path"] = yandex_links_path
        print(f"\nYandex-ссылки сохранены: {yandex_links_path}")

    # PDF-отчёт
    report_path = reports_dir / "geoplanner_report.pdf"
    generate_pdf_report(results=results, config=config, output_path=report_path)
    print(f"PDF-отчёт сохранён: {report_path}")

    # Финальная сводка
    print("\n" + "="*70)
    print("ИТОГОВАЯ СВОДКА ПО МЕНЕДЖЕРАМ")
    print("="*70)
    for mid, data in results["managers"].items():
        routes = data["routes"]
        used_days = data["used_days"]  # ← Используем правильное значение!
        print(f"Менеджер {mid}:")
        print(f"  Запланировано визитов: {data['visits_planned']}")
        print(f"  Дистанция: {data['distance_km']:.1f} км")
        print(f"  Использовано дней: {used_days}")
        if used_days > 0:
            print(f"  Средняя нагрузка: {data['visits_planned'] / used_days:.1f} визитов/день")

    print("\n" + "="*70)
    print("ГЛОБАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("="*70)
    total_managers = len(results["managers"])
    total_planned = results["total_visits_planned"]
    total_distance = results["total_distance_km"]
    excluded_count = len(results["excluded_points"])

    # === ГЛОБАЛЬНЫЙ ПОДСЧЁТ ИСПОЛЬЗОВАННЫХ ДНЕЙ ===
    all_assigned_days = set()
    for data in results["managers"].values():
        for route in data["routes"]:
            if route.workday_numbers:
                all_assigned_days.add(route.workday_numbers[0])
    global_used_days = len(all_assigned_days)

    print(f"Менеджеров обработано: {total_managers}")
    print(f"Всего запланировано визитов: {total_planned}")
    print(f"Общая дистанция: {total_distance:.1f} км")
    print(f"Использовано рабочих дней (глобально): {global_used_days}")
    print(f"Отсеяно точек: {excluded_count}")
    print(f"Карты сохранены в: {maps_dir}")
    print(f"Отчёт: {report_path}")
    if results["yandex_links_path"]:
        print(f"Yandex-ссылки: {results['yandex_links_path']}")
    print("="*70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Геопланировщик маршрутов менеджеров")
    parser.add_argument("--data", type=str, help="Путь к CSV с данными")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Путь к конфигу")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод в стратегиях")

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        config_path=args.config,
        verbose=args.verbose
    )