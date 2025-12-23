# app.py (Streamlit app with manager selection)

import streamlit as st
from datetime import datetime
from pathlib import Path
import tempfile
import os
import pandas as pd
from src.main import run_pipeline, generate_pdf_report

st.set_page_config(page_title="Геопланировщик", page_icon="🗺️", layout="wide")

# === Заголовок ===
st.title("🗺️ Геопланировщик визитов")
st.caption(f"Версия 1.0 | Дата: {datetime.now().strftime('%Y-%m-%d')}")

# === Загрузка данных ===
uploaded_file = st.file_uploader("📂 Загрузите CSV-файл с данными (point_id, manager, lat, lon, n_visits)", type="csv")

if uploaded_file:
    # Временный файл для данных
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_csv_path = tmp.name

    # Загрузка данных для выбора менеджеров
    df = pd.read_csv(temp_csv_path)
    all_managers = sorted(df['manager'].unique())
    selected_managers = st.multiselect(
        "Выберите менеджеров для оптимизации", 
        options=all_managers, 
        default=all_managers,
        help="Выберите один или несколько менеджеров. Если ничего не выбрать — обработаются все."
    )

    if not selected_managers:
        selected_managers = all_managers  # Если ничего не выбрано — все

# === Параметры ===
st.sidebar.header("⚙️ Параметры оптимизации")

col1, col2 = st.sidebar.columns(2)
with col1:
    max_points_per_cluster = st.sidebar.slider("Макс. точек в день", 8, 25, 18)
    max_visits_per_day = st.sidebar.slider("Макс. визитов в день", 8, 25, 12)
with col2:
    use_multi_day = st.sidebar.checkbox("Разбивать маршруты на несколько дней", value=True)

exclusion_percent = st.sidebar.slider("Отсев дальних точек (%)", 0, 50, 25)

n_clusters_base = st.sidebar.slider("Базовое кол-во дней (кластеров) на менеджера", 18, 35, 22)

st.sidebar.markdown("---")
run_button = st.sidebar.button("🚀 Запустить оптимизацию", type="primary", use_container_width=True)

# === Основная логика ===
if uploaded_file is None:
    st.info("👆 Пожалуйста, загрузите CSV-файл с данными для начала работы")
    st.stop()

# === Запуск пайплайна ===
if run_button:
    with st.spinner("🔄 Выполняется оптимизация..."):
        config_override = {
            "clustering": {
                "n_clusters_per_manager": {str(i): n_clusters_base for i in range(10)},
                "max_points_per_cluster": max_points_per_cluster,
                "max_visits_per_cluster": max_visits_per_day,
                "min_visits_per_cluster": max(6, max_visits_per_day - 4),
            },
            "routing": {
                "vehicle": {
                    "type": "car"
                },
                "use_split_route": use_multi_day,
                "max_daily_distance_km": 160
            },
            "exclusion": {
                "max_exclusion_percent": exclusion_percent
            },
            "scheduling": {
                "daily_distribution": {
                    "target_visits_per_day": max_visits_per_day,
                    "min_visits_per_day": max(8, max_visits_per_day - 4),
                    "max_visits_per_day": max_visits_per_day
                },
                "repeat_gap_days": 14
            },
            "data_processing": {
                "selected_managers": selected_managers  # ← НОВОЕ: передаём выбранных менеджеров в конфиг
            }
        }

        try:
            results = run_pipeline(
                data_path=temp_csv_path,
                config_path="configs/default_config.yaml",
                verbose=True,
                config_override=config_override
            )
            st.session_state.results = results
            st.session_state.temp_csv_path = temp_csv_path

            # Генерируем PDF
            report_path = Path("outputs/reports/geoplanner_report_streamlit.pdf")
            generate_pdf_report(results=results, config=config_override, output_path=report_path)
            st.session_state.report_path = str(report_path)

            st.success("✅ Оптимизация успешно завершена!")
        except Exception as e:
            st.error(f"Ошибка при выполнении пайплайна: {e}")
            import traceback
            st.code(traceback.format_exc())

# === Отображение результатов ===
if "results" in st.session_state:
    results = st.session_state.results

    st.markdown("## 📊 Ключевые результаты")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Менеджеров", len(results["managers"]))
    col2.metric("Запланировано визитов", results["total_visits_planned"])
    col3.metric("Общая дистанция", f"{results['total_distance_km']:.1f} км")
    col4.metric("Отсеяно точек", len(results["excluded_points"]))

    # Детализация по менеджерам
    st.markdown("### По менеджерам")
    manager_summary = []
    for mid, data in results["managers"].items():
        routes = data["routes"]
        used_days = len(set(r.workday_numbers[0] for r in routes if r.workday_numbers))
        manager_summary.append({
            "Менеджер": mid,
            "Дней с маршрутами": used_days,
            "Визитов": data["visits_planned"],
            "Дистанция (км)": round(data["distance_km"], 1),
            "Средняя нагрузка": round(data["visits_planned"] / used_days, 1) if used_days else 0
        })
    st.dataframe(pd.DataFrame(manager_summary), width='stretch')  # ← Исправлено: width='stretch'

    st.markdown("---")
    st.subheader("🗺️ Интерактивные карты маршрутов")

    if results["managers"]:
        tabs = st.tabs([f"Менеджер {mid}" for mid in sorted(results["managers"].keys())])
        for tab, mid in zip(tabs, sorted(results["managers"].keys())):
            with tab:
                map_path = results["map_paths"].get(mid)
                if map_path and Path(map_path).exists():
                    with open(map_path, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=750, scrolling=True)
                else:
                    st.warning(f"Карта для менеджера {mid} не найдена")

    st.markdown("---")
    st.subheader("🔗 Яндекс.Карты — готовые маршруты")

    if results.get("yandex_links_path"):
        yandex_path = Path(results["yandex_links_path"])
        if yandex_path.exists():
            df_yandex = pd.read_excel(yandex_path)
            st.dataframe(df_yandex, width='stretch')  # ← Исправлено: width='stretch'
            with open(yandex_path, "rb") as f:
                st.download_button(
                    "📥 Скачать таблицу ссылок (Excel)",
                    f,
                    file_name="yandex_routes_links.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    st.markdown("---")
    st.subheader("📄 Отчёт в PDF")

    if st.session_state.get("report_path"):
        report_path = Path(st.session_state.report_path)
        if report_path.exists():
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Скачать полный отчёт (PDF)",
                    f,
                    file_name=f"geoplanner_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    type="primary"
                )
        else:
            st.info("PDF-отчёт не найден")
    else:
        st.info("Запустите оптимизацию для генерации отчёта")

# === Очистка временных файлов ===
if st.session_state.get("temp_csv_path"):
    try:
        os.unlink(st.session_state.temp_csv_path)
        st.session_state.pop("temp_csv_path", None)
    except:
        pass

st.caption("Геопланировщик v1.0 • Декабрь 2025 • Оптимизация маршрутов с ИИ")