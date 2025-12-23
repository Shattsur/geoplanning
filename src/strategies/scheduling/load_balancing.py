# src/strategies/scheduling/load_balancing.py

"""Стратегия балансировки нагрузки по рабочим дням."""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from src.core.interfaces import SchedulingStrategy, Route

class LoadBalancingScheduling(SchedulingStrategy):
    """
    Распределяет готовые маршруты (Route) по рабочим дням с балансировкой нагрузки.
    Учитывает:
    - min/max/target визитов в день
    - минимальный интервал между повторными визитами одной точки (n_visits > 1)
    - возможность назначения повторного визита в тот же день при высокой нагрузке
    """

    def schedule(
        self,
        routes: List[Route],
        working_days: List[datetime],
        config: Dict[str, Any]
    ) -> List[Route]:
        if not routes or not working_days:
            return []

        scheduling_cfg = config.get("scheduling", {})
        daily_cfg = scheduling_cfg.get("daily_distribution", {})
        gap_days = scheduling_cfg.get("repeat_gap_days", 10)  # Уменьшено до 10 (из конфига)
        min_visits = daily_cfg.get("min_visits_per_day", 12)  # Увеличено до 12
        max_visits = daily_cfg.get("max_visits_per_day", 25)  # Увеличено до 25
        target_visits = daily_cfg.get("target_visits_per_day", 22)  # Увеличено до 22

        print(f"\n{'-'*60}")
        print("НАЗНАЧЕНИЕ РАБОЧИХ ДНЕЙ")
        print(f"{'-'*60}")
        print(f"Доступно рабочих дней: {len(working_days)} "
              f"(с {working_days[0].strftime('%d.%m')} по {working_days[-1].strftime('%d.%m')})")
        print(f"Ограничения: {min_visits} ≤ визитов/день ≤ {max_visits} (цель: ~{target_visits})")
        print(f"Мин. интервал между повторными визитами: {gap_days} дней")
        print(f"Маршрутов на входе: {len(routes)}")

        # Сортируем маршруты по нагрузке (визиты) — сначала тяжёлые
        sorted_routes = sorted(routes, key=lambda r: r.total_visits, reverse=True)

        # Текущая нагрузка по дням
        day_loads = [0] * len(working_days)
        day_routes: List[List[Route]] = [[] for _ in working_days]

        # === Шаг 1: Назначаем основные (первые) визиты ===
        print(f"\nНазначение основных визитов...")
        assigned_main: List[Route] = []

        for route in sorted_routes:
            candidates = [
                i for i, load in enumerate(day_loads)
                if load + route.total_visits <= max_visits
            ]

            if not candidates:
                print(f"  ⚠️ Маршрут {route.route_id} ({route.total_visits} визитов) не влез ни в один день!")
                continue

            # Выбираем день с минимальной текущей нагрузкой (для баланса)
            best_idx = min(candidates, key=lambda i: day_loads[i])
            best_day = working_days[best_idx]

            route.workday_numbers = (best_day.strftime("%Y-%m-%d"),)
            day_loads[best_idx] += route.total_visits
            day_routes[best_idx].append(route)
            assigned_main.append(route)

            print(f"  → {best_day.strftime('%d.%m')} | +{route.total_visits} виз. | "
                  f"итого: {day_loads[best_idx]} | маршрут: {len(route.points)} точек")

        # === Шаг 2: Назначаем повторные визиты (n_visits > 1) ===
        print(f"\nНазначение повторных визитов...")
        final_routes: List[Route] = assigned_main.copy()
        repeat_count = 0
        skipped_repeat = 0

        for route in assigned_main:
            if route.total_visits <= 1:
                continue

            repeat_route = route.copy()
            repeat_route.workday_numbers = ()
            day1_str = route.workday_numbers[0]
            day1_idx = next(i for i, d in enumerate(working_days) if d.strftime("%Y-%m-%d") == day1_str)
            min_day_idx = min(day1_idx + gap_days, len(working_days) - 1)
            found = False

            # Сначала ищем в отдельный день (как раньше)
            for candidate_idx in range(min_day_idx, len(working_days)):
                if day_loads[candidate_idx] + route.total_visits <= max_visits:
                    day = working_days[candidate_idx]
                    repeat_route.workday_numbers = (day.strftime("%Y-%m-%d"),)
                    day_loads[candidate_idx] += route.total_visits
                    day_routes[candidate_idx].append(repeat_route)
                    final_routes.append(repeat_route)
                    print(f"  → Повтор: {day.strftime('%d.%m')} | +{route.total_visits} виз. | "
                          f"итого: {day_loads[candidate_idx]}")
                    repeat_count += 1
                    found = True
                    break

            # Если не нашли — пытаемся назначить в ТОТ ЖЕ ДЕНЬ (если влезает)
            if not found:
                if day_loads[day1_idx] + route.total_visits <= max_visits:
                    repeat_route.workday_numbers = (day1_str,)
                    day_loads[day1_idx] += route.total_visits
                    day_routes[day1_idx].append(repeat_route)
                    final_routes.append(repeat_route)
                    print(f"  → Повтор в тот же день: {working_days[day1_idx].strftime('%d.%m')} | +{route.total_visits} виз.")
                    repeat_count += 1
                    found = True

            if not found:
                print(f"  ⚠️ Не удалось назначить повторный визит для маршрута {route.route_id}")
                skipped_repeat += 1

        # === Финальная статистика ===
        print(f"\n{'-'*60}")
        print("РЕЗУЛЬТАТ РАСПРЕДЕЛЕНИЯ ПО ДНЯМ")
        print(f"{'-'*60}")

        total_planned = 0
        used_days = 0
        for i, day in enumerate(working_days):
            load = day_loads[i]
            routes_count = len(day_routes[i])
            if load > 0:
                used_days += 1
                total_planned += load
                status = "✓" if min_visits <= load <= max_visits else "⚠"
                print(f"{status} {day.strftime('%d.%m (%a)'):12} | {load:2d} визитов | {routes_count} маршрут(ов)")
            else:
                print(f"  {day.strftime('%d.%m (%a)'):12} | {'—':10} | пустой день")

        avg_load = total_planned / used_days if used_days > 0 else 0
        print(f"\nИтог:")
        print(f"  • Использовано дней: {used_days} из {len(working_days)}")
        print(f"  • Всего запланировано визитов: {total_planned}")
        print(f"  • Средняя нагрузка: {avg_load:.1f} визитов/день")
        print(f"  • Повторных визитов назначено: {repeat_count}, пропущено: {skipped_repeat}")

        # Передаём правильное количество использованных дней в main.py через атрибут
        self.used_days_count = used_days  # ← НОВОЕ: для использования в main.py

        return final_routes