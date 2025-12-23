# src/visualization/map_visualizer.py

"""Визуализация маршрутов на Folium-картах с реальными дорожными линиями из OSRM."""

import folium
from folium.plugins import MarkerCluster
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import pandas as pd

from src.core.interfaces import Point, Depot, Route, Cluster


# Цвета для разных дней
DAY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

# Цвета для типа визита
VISIT_COLORS = {
    1: "blue",      # один визит
    2: "orange",    # два визита
    3: "red",       # три и более
}


def create_manager_map(
    routes: List[Route],
    depot: Depot,
    clusters: Optional[List[Cluster]] = None,
    excluded_points: Optional[List[Point]] = None,
    manager_id: int = 0,
    center: Tuple[float, float] = (56.32187, 43.94607),
    zoom_start: int = 11
) -> folium.Map:
    """
    Создаёт интерактивную Folium-карту с улучшенной визуализацией:
    - Разные цвета для количества визитов
    - Отметка повторных визитов
    - Чёткая легенда
    - Реальные дороги (OSRM) или fallback
    """
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")

    # === Депо ===
    folium.Marker(
        location=[depot.lat, depot.lon],
        popup=f"<b>ДЕПО: {depot.name}</b><br>Менеджер {manager_id}<br>{depot.lat:.6f}, {depot.lon:.6f}",
        icon=folium.Icon(color="black", icon="home", prefix="fa")
    ).add_to(m)

    # === Отсеянные точки ===
    if excluded_points:
        excluded_group = folium.FeatureGroup(name="❌ Отсеянные точки").add_to(m)
        for p in excluded_points:
            folium.CircleMarker(
                location=[p.lat, p.lon],
                radius=8,
                color="red",
                weight=2,
                fill=True,
                fill_opacity=0.7,
                popup=f"<b>ОТКЛОНЕНА</b><br>ID: {p.point_id}<br>Визитов: {p.n_visits}<br>Менеджер: {p.manager}"
            ).add_to(excluded_group)

    # === Кластер маркеров для точек ===
    marker_cluster = MarkerCluster(name="📍 Точки посещения").add_to(m)

    # Группируем маршруты по дате
    routes_by_day = {}
    for route in routes:
        day_str = route.workday_numbers[0] if route.workday_numbers else "Без даты"
        routes_by_day.setdefault(day_str, []).append(route)

    # === Отрисовка маршрутов по дням ===
    for day_str, day_routes in sorted(routes_by_day.items()):
        color_idx = hash(day_str) % len(DAY_COLORS)
        line_color = DAY_COLORS[color_idx]

        day_group = folium.FeatureGroup(name=f"📅 {day_str} ({len(day_routes)} маршрут(ов))").add_to(m)

        for route in day_routes:
            # Геометрия маршрута
            if route.geometry and len(route.geometry) > 2:
                poly_coords = route.geometry
                weight, opacity, dash = 8, 0.9, None
                road_type = "реальная дорога (OSRM)"
            else:
                # Fallback: депо → точки → депо
                poly_coords = [(depot.lat, depot.lon)]
                poly_coords.extend([(p.lat, p.lon) for p in route.points])
                poly_coords.append((depot.lat, depot.lon))
                weight, opacity, dash = 5, 0.6, '10'
                road_type = "прямые линии (fallback)"

            # Линия маршрута
            folium.PolyLine(
                locations=poly_coords,
                color=line_color,
                weight=weight,
                opacity=opacity,
                dash_array=dash,
                popup=folium.Popup(
                    f"<b>Маршрут {route.route_id}</b><br>"
                    f"Дата: <b>{day_str}</b><br>"
                    f"Точек: {len(route.points)} | Визитов: {route.total_visits}<br>"
                    f"Дистанция: <b>{route.total_distance_km:.1f} км</b><br>"
                    f"Время в пути: {route.estimated_driving_hours:.1f} ч<br>"
                    f"Тип: {road_type}",
                    max_width=300
                )
            ).add_to(day_group)

            # Точки маршрута
            for order_num, point in enumerate(route.points, 1):
                visit_count = point.n_visits
                icon_color = VISIT_COLORS.get(visit_count, "red")

                # Основной маркер с информацией
                folium.Marker(
                    location=[point.lat, point.lon],
                    popup=folium.Popup(
                        f"<b>{day_str} | Порядок: {order_num}</b><br>"
                        f"ID: {point.point_id}<br>"
                        f"Визитов всего: <b>{visit_count}</b><br>"
                        f"Это {'повторный' if visit_count > 1 else 'первый'} визит<br>"
                        f"Менеджер: {point.manager}<br>"
                        f"Маршрут: {route.route_id}",
                        max_width=300
                    ),
                    icon=folium.Icon(color=icon_color, icon="circle", prefix="fa")
                ).add_to(marker_cluster)

                # Номер на карте (только один раз на точку в этот день!)
                folium.Marker(
                    location=[point.lat, point.lon],
                    icon=folium.DivIcon(
                        icon_size=(32, 32),
                        icon_anchor=(16, 16),
                        html=f'''
                        <div style="
                            font-size: 13pt;
                            font-weight: bold;
                            color: white;
                            background: {line_color};
                            border: 3px solid white;
                            border-radius: 50%;
                            width: 32px; height: 32px;
                            text-align: center;
                            line-height: 32px;
                            box-shadow: 0 0 6px rgba(0,0,0,0.5);
                        ">{order_num}</div>
                        '''
                    )
                ).add_to(day_group)

            # Ночёвка (командировка)
            if route.is_multi_day and route.points:
                centroid_lat = np.mean([p.lat for p in route.points])
                centroid_lon = np.mean([p.lon for p in route.points])
                folium.Marker(
                    location=[centroid_lat, centroid_lon],
                    popup=f"<b>НОЧЁВКА</b><br>Маршрут: {route.route_id}<br>Дата: {day_str}",
                    icon=folium.Icon(color="purple", icon="bed", prefix="fa")
                ).add_to(day_group)

    # === Улучшенная легенда ===
    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; 
        width: 340px; 
        border:2px solid grey; 
        z-index:9999; 
        font-size:14px; 
        background:white; 
        padding:12px; 
        border-radius:12px; 
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
        line-height: 1.6;
    ">
      <b>🌍 Легенда — Менеджер {manager_id}</b><hr style="margin:8px 0;">
      <i class="fa fa-home" style="color:black"></i> Депо: {depot_name}<br>
      <i class="fa fa-circle" style="color:blue"></i> 1 визит<br>
      <i class="fa fa-circle" style="color:orange"></i> 2 визита<br>
      <i class="fa fa-circle" style="color:red"></i> ≥3 визита<br>
      <i class="fa fa-circle" style="color:red"></i> Отсеянная точка<br>
      <i class="fa fa-bed" style="color:purple"></i> Ночёвка (командировка)<br><br>
      <b>Сплошная линия</b> — реальная дорога (OSRM)<br>
      <b>Пунктир</b> — прямые линии (fallback)<br>
      <b>Разные цвета</b> — разные дни
    </div>
    '''.format(manager_id=manager_id, depot_name=depot.name)

    m.get_root().html.add_child(folium.Element(legend_html))

    # Управление слоями
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def save_folium_map(
    routes: List[Route],
    depot: Depot,
    output_path: Path | str,
    clusters: Optional[List[Cluster]] = None,
    excluded_points: Optional[List[Point]] = None,
    manager_id: int = 0
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    m = create_manager_map(
        routes=routes,
        depot=depot,
        clusters=clusters,
        excluded_points=excluded_points,
        manager_id=manager_id
    )
    m.save(str(output_path))
    print(f"Карта сохранена: {output_path}")


def generate_yandex_links(
    routes: List[Route],
    depots_by_manager: Dict[int, Depot],
    output_path: Path | str
) -> None:
    records = []

    for route in routes:
        manager = route.manager
        depot = depots_by_manager.get(manager)
        if not depot:
            continue

        day_str = route.workday_numbers[0] if route.workday_numbers else "Без даты"

        # Формируем координаты: депо → точки → депо
        coords_parts = [f"{depot.lon},{depot.lat}"]
        for p in route.points:
            coords_parts.append(f"{p.lon},{p.lat}")
        coords_parts.append(f"{depot.lon},{depot.lat}")

        yandex_url = "https://yandex.ru/maps/?rtext=" + "~".join(coords_parts) + "&rtt=auto"

        records.append({
            "Менеджер": manager,
            "Дата": day_str,
            "Маршрут ID": route.route_id,
            "Точек": len(route.points),
            "Визитов": route.total_visits,
            "Дистанция (км)": round(route.total_distance_km, 1),
            "Время в пути (ч)": round(route.estimated_driving_hours, 1),
            "Депо": depot.name,
            "Повторный визит": "Да" if any(p.n_visits > 1 for p in route.points) else "Нет",
            "Ссылка на Яндекс.Карты": yandex_url
        })

    if not records:
        print("Нет маршрутов для генерации Yandex-ссылок")
        return

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Yandex-ссылки сохранены: {output_path} ({len(df)} строк)")