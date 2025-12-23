# src/utils/distance.py 

"""Утилиты для расчёта расстояний и матриц расстояний.

Поддерживает только реальные дорожные расстояния через локальный OSRM 
"""

import numpy as np
import requests
from typing import List, Tuple, Optional

class OSRMLocalCalculator:
    """Калькулятор матрицы расстояний и геометрии через локальный OSRM (без fallback)."""
    
    def __init__(self):
        self.table_url = "http://localhost:5000/table/v1/driving"
        self.route_url = "http://localhost:5000/route/v1/driving"
        self.chunk_size = 80

    def _query_table_chunk(self, sources: List[Tuple[float, float]], destinations: List[Tuple[float, float]]) -> np.ndarray:
        coords = sources + destinations
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
            return np.array(distances) / 1000.0
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

            if np.all(matrix == 0):
                raise RuntimeError("Все расстояния от OSRM = 0 км — сервер не работает или данные не загружены")
            return matrix
        except Exception as e:
            raise RuntimeError(f"Не удалось получить матрицу расстояний от OSRM: {e}")
    
    def get_route_geometry(self, locations: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        if len(locations) < 2:
            return None

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
            return [(lat, lon) for lon, lat in coords]
        except Exception as e:
            print(f"Ошибка получения геометрии от OSRM: {e}")
            return None
        
def haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """
    Векторизованное haversine-расстояние (км).

    lat1, lon1: (N,)
    lat2, lon2: (M,) или (1,)
    Возвращает матрицу (N, M)
    """
    R = 6371.0  # радиус Земли, км

    lat1 = np.radians(lat1).reshape(-1, 1)
    lon1 = np.radians(lon1).reshape(-1, 1)
    lat2 = np.radians(lat2).reshape(1, -1)
    lon2 = np.radians(lon2).reshape(1, -1)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c