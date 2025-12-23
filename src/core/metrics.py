# src/core/metrics.py
"""
Ключевые метрики качества кластеризации и маршрутов.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from typing import Sequence, Any

from src.utils.distance import pairwise_haversine


def spatial_silhouette(
    points: np.ndarray,
    labels: np.ndarray,
    driving_factor: float = 1.2
) -> float:
    """Силуэт с учётом дорожных расстояний. От -1 до 1, выше — лучше."""
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
        return float("nan")

    dist_matrix = pairwise_haversine(points, driving_factor=driving_factor)
    return float(silhouette_score(dist_matrix, labels, metric="precomputed"))


def compactness_index(
    points: np.ndarray,
    labels: np.ndarray,
    driving_factor: float = 1.2
) -> float:
    """
    Средний диаметр кластера в км (меньше — лучше).
    Для кластера из 1 точки — диаметр = 0.
    """
    points = np.asarray(points)
    labels = np.asarray(labels)

    diameters = []
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        cluster_pts = points[labels == lbl]
        if len(cluster_pts) == 0:
            continue
        elif len(cluster_pts) == 1:
            diameters.append(0.0)
        else:
            dists = pairwise_haversine(cluster_pts, driving_factor=driving_factor)
            diameters.append(float(dists.max()))

    return float(np.mean(diameters)) if diameters else 0.0


def max_cluster_load(
    data: Sequence[dict] | pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    visits_col: str = "n_visits"
) -> int:
    """
    Максимальное количество визитов в одном кластере.
    Принимает:
      - pd.DataFrame
      - list[dict]
      - np.ndarray с именованными полями (structured array)
    """
    labels = np.asarray(labels)

    # Извлекаем массив визитов
    if isinstance(data, pd.DataFrame):
        visits = data[visits_col].values
    elif isinstance(data, (list, tuple, np.ndarray)):
        # list[dict] или structured array
        try:
            visits = np.array([item[visits_col] if isinstance(item, dict) else item for item in data])
        except (KeyError, IndexError):
            raise ValueError(f"Не удалось извлечь колонку '{visits_col}' из данных")
    else:
        raise TypeError("data должен быть DataFrame, list[dict] или ndarray")

    loads = [
        int(visits[labels == lbl].sum())
        for lbl in np.unique(labels)
        if lbl != -1 and np.sum(labels == lbl) > 0
    ]
    return max(loads) if loads else 0


# Тест при запуске напрямую
if __name__ == "__main__":
    points = np.array([
        [55.75, 37.62],
        [55.76, 37.63],
        [55.80, 37.65],
        [56.00, 37.70],
    ])
    labels = np.array([0, 0, 0, 1])

    print("Silhouette:", spatial_silhouette(points, labels))
    print("Compactness (km):", compactness_index(points, labels))

    test_data = pd.DataFrame({"n_visits": [1, 2, 1, 10]})
    print("Max load:", max_cluster_load(test_data, labels))  # Должно быть 10