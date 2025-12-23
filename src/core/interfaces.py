# src/core/interfaces.py

"""Абстрактные интерфейсы и базовые dataclasses для геопланировщика."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import numpy as np


class RouteStatus(Enum):
    PLANNED = "planned"
    FEASIBLE = "feasible"
    OVERLOADED = "overloaded"
    MULTI_DAY = "multi_day"  # командировка


@dataclass(frozen=True)
class Point:
    """Точка обслуживания (клиент/магазин)."""
    point_id: str
    manager: int
    lat: float
    lon: float
    n_visits: int = 1  # количество посещений в месяц


@dataclass(frozen=True)
class Depot:
    """Стартовая/конечная точка (офис/депо менеджера)."""
    lat: float
    lon: float
    name: str = "Depot"  # Используется в картах и отчётах (например, "Московский вокзал")


@dataclass
class Cluster:
    """Кластер точек для одного дня."""
    cluster_id: int
    points: Tuple[Point, ...]
    manager: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("Cluster не может быть пустым")
        managers = {p.manager for p in self.points}
        if len(managers) != 1:
            raise ValueError(f"В кластере несколько менеджеров: {managers}")
        self.manager = next(iter(managers))

    @property
    def total_visits(self) -> int:
        return sum(p.n_visits for p in self.points)

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class Route:
    """Маршрут на один или несколько дней."""
    route_id: str
    manager: int
    points: Tuple[Point, ...]
    depot: Depot
    workday_numbers: Tuple[str, ...] = field(default_factory=tuple)  # "YYYY-MM-DD"
    total_distance_km: float = 0.0
    estimated_driving_hours: float = 0.0
    estimated_service_hours: float = 0.0
    is_multi_day: bool = False
    status: RouteStatus = RouteStatus.PLANNED
    missed_points: List[Point] = field(default_factory=list)
    geometry: Optional[List[Tuple[float, float]]] = None  # ← НОВОЕ ПОЛЕ: реальная траектория от OSRM (lat, lon)

    @property
    def total_visits(self) -> int:
        return sum(p.n_visits for p in self.points)

    @property
    def total_hours(self) -> float:
        return self.estimated_driving_hours + self.estimated_service_hours

    def is_feasible(self, config: Dict[str, Any]) -> bool:
        routing_cfg = config.get("routing", {}).get("time", {})
        max_hours = routing_cfg.get("max_daily_hours", 9)
        service_min = routing_cfg.get("service_time_minutes", 30)

        service_hours = len(self.points) * service_min / 60.0
        total = self.estimated_driving_hours + service_hours

        if self.is_multi_day:
            max_hours *= 1.5  # больше времени в командировке

        return total <= max_hours

    def copy(self) -> "Route":
        """Копия для повторного визита (n_visits > 1)."""
        return Route(
            route_id=self.route_id + "_repeat",
            manager=self.manager,
            points=self.points,
            depot=self.depot,
            workday_numbers=self.workday_numbers,
            total_distance_km=self.total_distance_km,
            estimated_driving_hours=self.estimated_driving_hours,
            estimated_service_hours=self.estimated_service_hours,
            is_multi_day=self.is_multi_day,
            status=self.status,
            missed_points=self.missed_points[:],
            geometry=self.geometry  # ← копируем геометрию
        )


# === Абстрактные стратегии ===

class ExclusionStrategy(ABC):
    @abstractmethod
    def exclude(
        self,
        points: Sequence[Point],
        depot: Depot,
        config: Dict[str, Any]
    ) -> Tuple[List[Point], List[Point]]:
        """Возвращает (kept_points, excluded_points)."""
        pass


class ClusteringStrategy(ABC):
    @abstractmethod
    def fit_predict(
        self,
        points: Sequence[Point],
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Возвращает массив меток кластеров."""
        pass

    @abstractmethod
    def get_clusters(self) -> List[Cluster]:
        """Возвращает список готовых Cluster объектов."""
        pass


class RoutingStrategy(ABC):
    @abstractmethod
    def optimize(
        self,
        cluster: Cluster,
        depot: Depot,
        config: Dict[str, Any]
    ) -> Route:
        """Оптимизирует маршрут и возвращает Route с geometry (если возможно)."""
        pass


class SchedulingStrategy(ABC):
    @abstractmethod
    def schedule(
        self,
        routes: List[Route],
        working_days: List[datetime],
        config: Dict[str, Any]
    ) -> List[Route]:
        """Назначает дни маршрутам."""
        pass