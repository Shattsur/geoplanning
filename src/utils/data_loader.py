"""
Загрузка и предобработка данных.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from src.core.interfaces import Point
from src.core.validator import validate_input_data

load_dotenv()  # Загружаем .env один раз


# ============================================================================
# Загрузка данных
# ============================================================================

def load_raw_data(data_path: str | None = None) -> pd.DataFrame:
    """
    Загрузка сырых данных из CSV.
    """
    if data_path is None:
        data_path = os.getenv("DATA_PATH")
        if not data_path:
            raise ValueError("DATA_PATH не указан ни в параметрах, ни в .env")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {data_path}")

    df = pd.read_csv(data_path)

    expected_columns = {"point_id", "manager", "lat", "lon", "n_visits"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    # Приведение типов
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)
    df["manager"] = df["manager"].astype(int)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["n_visits"] = df["n_visits"].astype(int)

    print(f"Загружено строк: {len(df)}, уникальных point_id: {df['point_id'].nunique()}")
    return df


# ============================================================================
# Дубликаты
# ============================================================================

def handle_duplicates(df: pd.DataFrame, mode: str = "rename") -> pd.DataFrame:
    """
    Обработка дубликатов point_id внутри одного менеджера.

    Поддерживаемые режимы:
    - keep_first : оставить первую запись
    - rename     : переименовать дубликаты с суффиксом (_2, _3, ...)
    """
    if mode == "keep_first":
        print("Дубликаты: оставляем первую запись")
        return df.drop_duplicates(subset=["manager", "point_id"], keep="first")

    if mode == "rename":
        print("Дубликаты: переименование с суффиксами")
        df = df.copy()
        counts = df.groupby(["manager", "point_id"]).cumcount()
        df.loc[counts > 0, "point_id"] = (
            df.loc[counts > 0, "point_id"] + "_" + (counts[counts > 0] + 1).astype(str)
        )
        return df

    raise ValueError(f"Неизвестный режим обработки дубликатов: {mode}")


# ============================================================================
# Преобразование
# ============================================================================

def df_to_points(df: pd.DataFrame) -> List[Point]:
    """
    Преобразование DataFrame в список Point.
    """
    return [
        Point(
            point_id=row.point_id,
            manager=row.manager,
            lat=row.lat,
            lon=row.lon,
            n_visits=row.n_visits,
        )
        for row in df.itertuples(index=False)
    ]


# ============================================================================
# Основной вход
# ============================================================================

def load_and_prepare_data(
    config: Dict[str, Any],
    data_path: str | None = None
) -> Tuple[List[Point], Dict[int, Any]]:
    """
    Загружает и подготавливает данные.
    Возвращает:
      - список точек
      - пустой словарь депо (создаётся в main.py)
    """
    df = load_raw_data(data_path)

    # Фильтр по менеджерам
    selected_managers = config.get("data_processing", {}).get("selected_managers")
    if selected_managers:
        df = df[df["manager"].isin(selected_managers)]
        print(f"Фильтр по менеджерам: {selected_managers}, осталось строк: {len(df)}")

    # Валидация
    valid, errors, warnings = validate_input_data(df)
    for w in warnings:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {w}")
    if not valid:
        raise ValueError("Ошибки в данных: " + "; ".join(errors))

    # Дубликаты
    dup_mode = config.get("data_processing", {}).get("duplicate_mode", "rename")
    df = handle_duplicates(df, mode=dup_mode)

    # В точки
    points = df_to_points(df)
    print(f"Подготовлено точек: {len(points)}")

    return points, {}
