# src/strategies/clustering/kmeans_optimized.py
"""Capacitated K-Means с контролем целевого количества кластеров и балансировкой ограничений."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Tuple
from collections import Counter
from src.core.interfaces import ClusteringStrategy, Cluster, Point
from src.core.validator import check_cluster_constraints
from src.utils.data_loader import df_to_points
import warnings


class KMeansCapacitated(ClusteringStrategy):
    """
    Capacitated K-Means с фокусом на целевом количестве кластеров.
    Особенности:
    - Строгий контроль целевого количества кластеров (22 для декабря)
    - Пошаговая балансировка: сначала разбиваем перегруженные, потом объединяем маленькие
    - Гибкие ограничения с приоритетами: визиты > точки > диаметр
    - Автоматическая корректировка при превышении лимита кластеров

    Параметры конфигурации:
        max_visits_per_cluster: 25 (максимальная нагрузка в день)
        target_clusters: 22 (рабочие дни декабря)
        max_clusters_allowed: 26 (максимум для гибкости)
        min_points_per_cluster: 3 (минимум для объединения)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.labels_: np.ndarray | None = None
        self.points_df_: pd.DataFrame | None = None
        self.points_: List[Point] = []
        self.min_cluster_size: int = 3
        self.target_clusters: int = 22
        self.max_clusters_allowed: int = 26  # Максимум 118% от целевого
        
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
            
    def _get_violation_priority(self, violations: Dict[str, List[Tuple[int, Any]]]) -> Tuple[int, float, str]:
        """
        Определяет приоритет нарушений для выбора кластера на разбиение.
        Возвращает: (cluster_id, severity_score, violation_type)
        """
        priority_scores = {
            'overloaded_visits': 3.0,   # Наивысший приоритет
            'too_many_points': 2.5,
            'large_diameter': 1.0,
            'too_few_visits': -1.0,    # Низкий приоритет для объединения
            'too_few_points': -1.0
        }
        
        best_score = -float('inf')
        best_cluster = None
        best_type = None
        
        for viol_type, clusters in violations.items():
            if not clusters:
                continue
                
            weight = priority_scores.get(viol_type, 0.5)
            for cluster_id, value in clusters:
                # Степень нарушения: 1.0 = 100% превышения лимита
                if viol_type == 'overloaded_visits':
                    severity = (value - 25) / 25  # Нормализация по лимиту
                elif viol_type == 'too_many_points':
                    severity = (value - 35) / 35
                else:
                    severity = 0.5  # Для диаметра используем фиксированный вес
                    
                score = weight * max(0.1, severity)  # Минимальный вес 0.1
                
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
                    best_type = viol_type
                    
        return best_cluster, best_score, best_type
        
    def _merge_small_clusters(self, labels: np.ndarray, violations: Dict[str, List[Tuple[int, Any]]]) -> np.ndarray:
        """
        Объединяет маленькие кластеры для снижения общего количества.
        Выбирает пары кластеров с минимальным расстоянием между центроидами.
        """
        if 'too_few_points' not in violations or not violations['too_few_points']:
            return labels
            
        small_clusters = [cid for cid, _ in violations['too_few_points'] if _ <= 3]
        if len(small_clusters) < 2:
            return labels
            
        # Получаем центроиды всех кластеров
        centroids = {}
        for cid in np.unique(labels):
            mask = labels == cid
            if np.sum(mask) > 0:
                cluster_points = self.points_df_[['lat', 'lon']].values[mask]
                centroids[cid] = np.mean(cluster_points, axis=0)
                
        # Находим ближайшие пары маленьких кластеров
        merge_pairs = []
        used_clusters = set()
        
        for i, cid1 in enumerate(small_clusters):
            if cid1 in used_clusters:
                continue
                
            min_dist = float('inf')
            best_pair = None
            
            for cid2 in small_clusters[i+1:]:
                if cid2 in used_clusters:
                    continue
                    
                if cid1 not in centroids or cid2 not in centroids:
                    continue
                    
                dist = np.linalg.norm(centroids[cid1] - centroids[cid2])
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (cid1, cid2)
                    
            if best_pair and min_dist < 10.0:  # 10 км - разумный порог для объединения
                merge_pairs.append(best_pair)
                used_clusters.update(best_pair)
                
        # Выполняем объединение
        new_labels = labels.copy()
        for cid1, cid2 in merge_pairs:
            new_labels[new_labels == cid2] = cid1
            
        # Перенумеровываем кластеры
        unique_labels = np.unique(new_labels)
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        new_labels = np.vectorize(label_map.get)(new_labels)
        
        self._log(f"  → Объединено {len(merge_pairs)} пар маленьких кластеров")
        return new_labels
        
    def _split_overloaded_cluster(self, cluster_id: int, n_sub: int, coords: np.ndarray, visits: np.ndarray, labels: np.ndarray, random_state: int) -> np.ndarray:
        """
        Разбивает перегруженный кластер на n_sub подкластеров.
        """
        mask = labels == cluster_id
        sub_coords = coords[mask]
        sub_visits = visits[mask]
        
        if len(sub_coords) < n_sub:
            n_sub = max(2, len(sub_coords) // 2)
            
        sub_kmeans = KMeans(
            n_clusters=n_sub,
            n_init=20,
            max_iter=400,
            random_state=random_state,
            algorithm="lloyd"
        )
        
        sub_labels = sub_kmeans.fit_predict(sub_coords)
        new_base = labels.max() + 1
        new_labels = labels.copy()
        new_labels[mask] = new_base + sub_labels
        
        return new_labels
        
    def fit_predict(self, points: List[Point], config: Dict[str, Any]) -> np.ndarray:
        if not points:
            raise ValueError("Список точек пустой")
            
        # Сохраняем исходные точки
        self.points_ = points
        
        data = {
            "point_id": [p.point_id for p in points],
            "manager": [p.manager for p in points],
            "lat": [p.lat for p in points],
            "lon": [p.lon for p in points],
            "n_visits": [p.n_visits for p in points],
        }
        points_df = pd.DataFrame(data).reset_index(drop=True)
        self.points_df_ = points_df
        
        clustering_cfg = config.get("clustering", {})
        routing_cfg = config.get("routing", {}).get("time", {})
        manager = points[0].manager
        
        # Параметры кластеризации
        self.target_clusters = clustering_cfg.get("n_clusters_per_manager", {}).get(str(manager), 22)
        self.max_clusters_allowed = int(self.target_clusters * 1.18)  # Максимум 118% от цели
        
        max_visits = clustering_cfg.get("max_visits_per_cluster", 25)
        max_points = clustering_cfg.get("max_points_per_cluster", 35)
        min_points = clustering_cfg.get("min_points_per_cluster", 3)  # Снижено для гибкости
        min_visits = clustering_cfg.get("min_visits_per_cluster", 6)
        max_diameter = clustering_cfg.get("max_diameter_km", 100.0)
        driving_factor = routing_cfg.get("driving_factor", 1.3)
        random_state = config.get("project", {}).get("random_state", 42)
        
        coords = points_df[["lat", "lon"]].values
        visits = points_df["n_visits"].values
        
        self._log(f"\n{'='*20} КЛАСТЕРИЗАЦИЯ МЕНЕДЖЕР {manager} {'='*20}")
        self._log(f"Исходно: {len(points_df)} точек, {visits.sum()} визитов")
        self._log(f"Параметры: цель = {self.target_clusters} кластеров, максимум = {self.max_clusters_allowed}")
        self._log(f"Ограничения: ≤{max_visits} виз., ≤{max_points} точ., диаметр ≤{max_diameter} км")
        
        # === Этап 1: Начальная кластеризация с целевым числом кластеров ===
        kmeans = KMeans(
            n_clusters=self.target_clusters,
            n_init=30,
            max_iter=600,
            random_state=random_state,
            algorithm="lloyd"
        )
        labels = kmeans.fit_predict(coords)
        
        # === Этап 2: Итеративная корректировка с контролем количества кластеров ===
        iteration = 0
        max_iter = 50  # Максимум итераций для безопасности
        best_labels = labels.copy()
        best_violation_count = float('inf')
        
        while iteration < max_iter:
            iteration += 1
            current_clusters = len(np.unique(labels))
            
            # Проверяем ограничения
            violations = check_cluster_constraints(
                labels=labels,
                points_df=points_df,
                config={
                    "clustering": {
                        "max_visits_per_cluster": max_visits,
                        "max_points_per_cluster": max_points,
                        "min_points_per_cluster": min_points,
                        "min_visits_per_cluster": min_visits,
                        "max_diameter_km": max_diameter,
                    },
                    "routing": {"time": {"driving_factor": driving_factor}}
                }
            )
            
            # Считаем общее количество нарушений
            violation_count = sum(
                len(v) for k, v in violations.items() 
                if k in ['overloaded_visits', 'too_many_points', 'large_diameter']
            )
            
            # Отслеживаем лучшее решение
            if violation_count < best_violation_count or (
                violation_count == best_violation_count and current_clusters < len(np.unique(best_labels))
            ):
                best_violation_count = violation_count
                best_labels = labels.copy()
                
            # Если нет критических нарушений и количество кластеров в пределах нормы - завершаем
            if violation_count == 0 and current_clusters <= self.max_clusters_allowed:
                self._log(f"✓ Оптимальное решение найдено за {iteration} итераций!")
                break
                
            self._log(f"  Итерация {iteration}: кластеров = {current_clusters}, нарушений = {violation_count}")
            
            # === Шаг 1: Если кластеров слишком много - объединяем маленькие ===
            if current_clusters > self.max_clusters_allowed:
                self._log(f"  → Слишком много кластеров ({current_clusters} > {self.max_clusters_allowed})")
                labels = self._merge_small_clusters(labels, violations)
                continue
                
            # === Шаг 2: Если есть критические нарушения - разбиваем перегруженные кластеры ===
            cluster_to_fix, score, viol_type = self._get_violation_priority(violations)
            
            if cluster_to_fix is not None and score > 0:
                mask = labels == cluster_to_fix
                sub_n = np.sum(mask)
                current_visits = int(visits[mask].sum())
                
                # Определяем количество подкластеров
                if viol_type == "overloaded_visits":
                    n_sub = max(2, int(np.ceil(current_visits / max_visits)))
                elif viol_type == "too_many_points":
                    n_sub = max(2, int(np.ceil(sub_n / max_points)))
                else:  # large_diameter
                    n_sub = 2
                    
                # Ограничиваем максимальное количество подкластеров
                n_sub = min(n_sub, 3)  # Не более 3 подкластеров за раз
                future_clusters = current_clusters - 1 + n_sub
                
                if future_clusters <= self.max_clusters_allowed:
                    self._log(f"  → Разбиваем кластер {cluster_to_fix} ({viol_type}, score={score:.2f}): "
                            f"{sub_n} точек, {current_visits} визитов → {n_sub} подкластеров")
                    labels = self._split_overloaded_cluster(
                        cluster_id=cluster_to_fix,
                        n_sub=n_sub,
                        coords=coords,
                        visits=visits,
                        labels=labels,
                        random_state=random_state + iteration
                    )
                else:
                    self._log(f"  ⚠ Пропускаем разбиение кластера {cluster_to_fix} - превысит лимит кластеров")
            else:
                # Нет критических нарушений или дальнейшие разбиения невозможны
                if current_clusters > self.target_clusters:
                    self._log("  → Оптимизация завершена. Кластеров больше цели, но нет критических нарушений")
                break
                
        # Используем лучшее найденное решение
        labels = best_labels
        
        # === Финальная статистика ===
        unique_labels = np.unique(labels)
        final_clusters = len(unique_labels)
        
        if final_clusters > self.max_clusters_allowed:
            warnings.warn(f"Превышен лимит кластеров: {final_clusters} > {self.max_clusters_allowed}")
        
        self._log(f"✓ Кластеризация завершена: {final_clusters} кластеров (цель {self.target_clusters})")
        self._log(f"  • Мин. кластеров: {len([l for l in np.unique(labels) if np.sum(labels == l) >= min_points])}")
        self._log(f"  • Макс. визитов в кластере: {max(np.sum(points_df['n_visits'][labels == l]) for l in unique_labels)}")
        
        # Перенумерация от 0
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        labels = np.vectorize(label_map.get)(labels)
        self.labels_ = labels.astype(int)
        
        return self.labels_
        
    def get_clusters(self) -> List[Cluster]:
        """
        Преобразует результаты кластеризации в список объектов Cluster.
        Фильтрует шум (label=-1) и слишком маленькие кластеры.
        """
        if self.labels_ is None or self.points_ is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit_predict()")
        
        clusters = []
        unique_labels = np.unique(self.labels_)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # шум (DBSCAN) - игнорируем
                continue
            
            mask = self.labels_ == cluster_id
            cluster_points = [self.points_[i] for i in range(len(self.points_)) if mask[i]]
            
            if not cluster_points:
                continue
            
            # Проверяем минимальный размер кластера
            if len(cluster_points) < self.min_cluster_size:
                continue
            
            # Создаем кластер БЕЗ параметра manager в конструкторе
            # (согласно интерфейсу Cluster)
            clusters.append(Cluster(
                cluster_id=int(cluster_id),
                points=tuple(cluster_points)
                # manager убран из аргументов
            ))
        
        return clusters