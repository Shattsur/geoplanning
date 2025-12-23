# src/strategies/routing/ortools_vrp.py

"""Оптимизация маршрута внутри кластера с помощью OR-Tools + локальный OSRM (только реальные дороги)."""

from __future__ import annotations

import numpy as np
import requests
from typing import List, Tuple, Optional, Dict, Any
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.core.interfaces import RoutingStrategy, Route, Depot, Cluster


class OSRMLocalCalculator:
    """Калькулятор матрицы расстояний и геометрии через локальный OSRM (без fallback)."""
    
    def __init__(self):
        self.table_url = "http://localhost:5000/table/v1/driving"
        self.route_url = "http://localhost:5000/route/v1/driving"
        self.chunk_size = 80  # OSRM ограничивает ~100 точек на запрос

    def _query_table_chunk(self, sources: List[Tuple[float, float]], destinations: List[Tuple[float, float]]) -> np.ndarray:
        # OSRM ожидает координаты в формате (lon, lat)
        coords = sources + destinations
        # Преобразуем (lat, lon) → "lon,lat"
        coords_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)

        url = f"{self.table_url}/{coords_str}"

        params = {
            "annotations": "distance",
            "sources": ";".join(str(i) for i in range(len(sources))),
            "destinations": ";".join(str(len(sources) + i) for i in range(len(destinations)))
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(f"OSRM table error {response.status_code}: {response.text}")
            data = response.json()
            distances = data.get("distances")
            if distances is None:
                raise RuntimeError("OSRM вернул пустые distances")
            return np.array(distances) / 1000.0  # метры → км
        except Exception as e:
            raise RuntimeError(f"Ошибка запроса к OSRM table: {e}")

    def get_distance_matrix(self, points: List[Tuple[float, float]]) -> np.ndarray:
        n = len(points)
        if n <= 1:
            return np.zeros((n, n))

        matrix = np.zeros((n, n))
        try:
            for i_start in range(0, n, self.chunk_size):
                i_end = min(i_start + self.chunk_size, n)
                sources = points[i_start:i_end]

                for j_start in range(0, n, self.chunk_size):
                    j_end = min(j_start + self.chunk_size, n)
                    destinations = points[j_start:j_end]
                    chunk = self._query_table_chunk(sources, destinations)

                    for si, gi in enumerate(range(i_start, i_end)):
                        for dj, gj in enumerate(range(j_start, j_end)):
                            matrix[gi][gj] = chunk[si][dj]

            # Проверка: если все расстояния 0 — OSRM не работает
            if np.all(matrix == 0):
                raise RuntimeError("Все расстояния от OSRM = 0 км — сервер не работает или данные не загружены")
            return matrix
        except Exception as e:
            raise RuntimeError(f"Не удалось получить матрицу расстояний от OSRM: {e}")

    def get_route_geometry(self, locations: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        if len(locations) < 2:
            return None

        # OSRM ожидает "lon,lat"
        coords_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in locations)
        url = f"{self.route_url}/{coords_str}"

        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false"
        }

        try:
            response = requests.get(url, params=params, timeout=40)
            if response.status_code != 200:
                print(f"OSRM route warning {response.status_code}: {response.text}")
                return None

            data = response.json()
            routes = data.get("routes")
            if not routes:
                return None

            geometry = routes[0].get("geometry")
            if not geometry or geometry.get("type") != "LineString":
                return None

            coords = geometry["coordinates"]
            # OSRM возвращает [[lon, lat], ...] → преобразуем в [(lat, lon), ...] для Folium
            return [(lat, lon) for lon, lat in coords]
        except Exception as e:
            print(f"Ошибка получения геометрии от OSRM: {e}")
            return None


class ORToolsVRPStrategy(RoutingStrategy):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.osrm_calc = OSRMLocalCalculator()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def optimize(self, cluster: Cluster, depot: Depot, config: Dict[str, Any]) -> Route:
        points = list(cluster.points)
        if not points:
            return Route(
                route_id=f"empty_{cluster.cluster_id}",
                manager=cluster.manager,
                depot=depot,
                points=(),
                workday_numbers=(),
                geometry=None
            )

        time_cfg = config.get("routing", {}).get("time", {})
        service_minutes = time_cfg.get("service_time_minutes", 30)
        avg_speed_kmh = time_cfg.get("avg_speed_kmh", 40)

        # === Только депо → точки → депо (без виртуального отеля) ===
        locations = [(depot.lat, depot.lon)] + [(p.lat, p.lon) for p in points]

        self._log(f"Маршрутизация кластера {cluster.cluster_id}: {len(points)} точек")

        try:
            distance_matrix_km = self.osrm_calc.get_distance_matrix(locations)
            distance_matrix_m = (distance_matrix_km * 1000).astype(int)
            np.fill_diagonal(distance_matrix_m, 0)
        except Exception as e:
            self._log(f"Критическая ошибка OSRM: {e}")
            # Возвращаем пустой маршрут с предупреждением
            return Route(
                route_id=f"error_{cluster.cluster_id}",
                manager=cluster.manager,
                depot=depot,
                points=tuple(points),
                total_distance_km=0.0,
                estimated_driving_hours=0.0,
                estimated_service_hours=len(points) * service_minutes / 60.0,
                is_multi_day=False,
                workday_numbers=(),
                missed_points=[],
                geometry=None
            )

        n_nodes = len(locations)
        manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix_m[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Ограничение по дистанции
        max_distance_m = config.get("routing", {}).get("max_daily_distance_km", 250) * 1000
        routing.AddDimension(
            transit_callback_index,
            0,
            int(max_distance_m),
            True,
            "Distance"
        )
        routing.GetDimensionOrDie("Distance").SetGlobalSpanCostCoefficient(100)

        # Параметры поиска
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 60  # Увеличено для больших кластеров

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            order = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                order.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            order.append(0)
            self._log("OR-Tools нашёл оптимальный маршрут")
        else:
            self._log("OR-Tools не нашёл решение → ближайший сосед")
            # Nearest neighbor как запасной
            visited = np.full(n_nodes, False)
            order = [0]
            visited[0] = True
            current = 0
            while len(order) < n_nodes:
                dists = distance_matrix_m[current].copy()
                dists[visited] = 10**9
                next_node = int(np.argmin(dists))
                order.append(next_node)
                visited[next_node] = True
                current = next_node
            order.append(0)

        # Порядок точек (без депо)
        route_points = [points[i - 1] for i in order if 0 < i <= len(points)]

        # Дистанция и время
        total_km = sum(distance_matrix_m[order[i]][order[i + 1]] for i in range(len(order) - 1)) / 1000.0
        driving_hours = total_km / avg_speed_kmh if avg_speed_kmh > 0 else 0
        service_hours = len(route_points) * service_minutes / 60.0

        # Геометрия реальной дороги
        route_locations = [locations[i] for i in order]
        geometry = self.osrm_calc.get_route_geometry(route_locations)

        return Route(
            route_id=f"route_{cluster.cluster_id}",
            manager=cluster.manager,
            points=tuple(route_points),
            depot=depot,
            total_distance_km=round(total_km, 2),
            estimated_driving_hours=round(driving_hours, 2),
            estimated_service_hours=round(service_hours, 2),
            is_multi_day=False,  # Командировки отключены
            workday_numbers=(),
            missed_points=[],
            geometry=geometry
        )