# src/visualization/report_generator.py

"""Генерация профессионального PDF-отчёта с результатами геопланирования."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.core.interfaces import Route

# === Регистрация шрифта с поддержкой кириллицы ===
# Помести файл DejaVuSans.ttf в корень проекта или в папку fonts/
FONT_PATH = Path(__file__).parent.parent.parent / "DejaVuSans.ttf"
if FONT_PATH.exists():
    pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_PATH)))
    DEFAULT_FONT = "DejaVu"
else:
    # Fallback на Helvetica (без кириллицы — может дать "nnnnnn")
    DEFAULT_FONT = "Helvetica"
    print("Предупреждение: DejaVuSans.ttf не найден — возможны проблемы с кириллицей в PDF")


def _create_visits_histogram(routes: List[Route]) -> RLImage | None:
    visits = [r.total_visits for r in routes if r.total_visits > 0]
    if not visits:
        return None

    plt.figure(figsize=(8, 4))
    plt.hist(visits, bins=range(min(visits), max(visits) + 2), color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title("Распределение визитов по дням", fontsize=14, fontfamily=DEFAULT_FONT)
    plt.xlabel("Визитов в день")
    plt.ylabel("Количество дней")
    plt.grid(axis='y', alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return RLImage(buf, width=14*cm, height=8*cm)


def _create_distance_pie(routes: List[Route]) -> RLImage | None:
    managers_dist = {}
    for r in routes:
        managers_dist[r.manager] = managers_dist.get(r.manager, 0) + r.total_distance_km

    if len(managers_dist) <= 1:
        return None

    labels = [f"Менеджер {m}" for m in managers_dist.keys()]
    sizes = list(managers_dist.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%', startangle=90)
    plt.title("Распределение дистанции по менеджерам", fontsize=14, fontfamily=DEFAULT_FONT)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return RLImage(buf, width=12*cm, height=12*cm)


def _create_footer(canvas, doc):
    """Футер на каждой странице с правильной датой и версией."""
    canvas.saveState()
    canvas.setFont(DEFAULT_FONT, 9)
    footer_text = (
        f"Геопланировщик v1.0 • {datetime.now().strftime('%d %B %Y, %H:%M')} • "
        "Оптимизация маршрутов с ИИ"
    )
    canvas.drawCentredString(doc.pagesize[0] / 2, 30, footer_text)
    canvas.restoreState()


def generate_pdf_report(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path | str
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=3*cm,
        onPage=_create_footer  # ← Добавлен футер с правильной датой
    )

    styles = getSampleStyleSheet()

    # Стили с поддержкой кириллицы
    style_title = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName=DEFAULT_FONT,
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=30,
        textColor=HexColor("#1A3C5A")
    )
    style_h1 = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontName=DEFAULT_FONT,
        fontSize=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=HexColor("#2E4057")
    )
    style_h2 = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontName=DEFAULT_FONT,
        fontSize=14,
        spaceAfter=10,
        textColor=HexColor("#4A6FA5")
    )
    style_normal = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontName=DEFAULT_FONT,
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    style_center = ParagraphStyle(
        "CustomCenter",
        parent=style_normal,
        alignment=TA_CENTER
    )

    story = []

    # Титульный лист
    story.append(Paragraph("Отчёт по оптимизации маршрутов полевых менеджеров", style_title))
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph(f"Дата генерации: {datetime.now().strftime('%d %B %Y, %H:%M')}", style_center))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Геопланировщик v1.0", style_center))
    story.append(PageBreak())

    # Общая сводка
    story.append(Paragraph("Общая сводка", style_h1))

    total_visits_initial = results["total_visits_planned"] + sum(p.n_visits for p in results["excluded_points"])
    coverage = results["total_visits_planned"] / total_visits_initial if total_visits_initial else 0

    summary_data = [
        ["Показатель", "Значение", "Комментарий"],
        ["Исходное количество визитов", str(total_visits_initial), "Все точки из данных"],
        ["Запланировано визитов", f"{results['total_visits_planned']}", f"{coverage:.1%} от исходного"],
        ["Отсеяно визитов", str(total_visits_initial - results["total_visits_planned"]), "Дальние/низкоприоритетные"],
        ["Общая дистанция", f"{results['total_distance_km']:.1f} км", "По реальным дорогам (OSRM)"],
        ["Менеджеров", str(len(results["managers"])), ""]
    ]

    table = Table(summary_data, colWidths=[6*cm, 4*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1A3C5A")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), DEFAULT_FONT),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#AAAAAA")),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F8F9FA")),
    ]))
    story.append(table)
    story.append(Spacer(1, 1*cm))

    all_routes = [r for m in results["managers"].values() for r in m.get("routes", [])]

    hist = _create_visits_histogram(all_routes)
    if hist:
        story.append(Paragraph("Распределение визитов по дням", style_h2))
        story.append(KeepTogether(hist))
        story.append(Spacer(1, 1*cm))

    pie = _create_distance_pie(all_routes)
    if pie:
        story.append(Paragraph("Распределение дистанции по менеджерам", style_h2))
        story.append(KeepTogether(pie))
        story.append(Spacer(1, 1*cm))

    # По менеджерам
    story.append(Paragraph("Детализация по менеджерам", style_h1))

    for manager_id, data in results["managers"].items():
        routes = data.get("routes", [])
        depot = data.get("depot")

        story.append(Paragraph(f"Менеджер {manager_id} — {depot.name if depot else 'Без названия'}", style_h2))

        used_days = len(set(r.workday_numbers[0] for r in routes if r.workday_numbers))
        mgr_data = [
            ["Показатель", "Значение"],
            ["Депо", depot.name if depot else "—"],
            ["Дней с маршрутами", str(used_days)],
            ["Запланировано визитов", str(sum(r.total_visits for r in routes))],
            ["Общая дистанция", f"{sum(r.total_distance_km for r in routes):.1f} км"],
            ["Средняя нагрузка", f"{sum(r.total_visits for r in routes) / used_days:.1f} виз./день" if used_days else "—"]
        ]

        mgr_table = Table(mgr_data, colWidths=[8*cm, 8*cm])
        mgr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#4A6FA5")),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ('FONTNAME', (0, 0), (-1, -1), DEFAULT_FONT),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F0F4F8")),
        ]))
        story.append(mgr_table)
        story.append(Spacer(1, 0.5*cm))

        # Карта (если есть скриншот)
        map_path = results["map_paths"].get(manager_id)
        if map_path:
            screenshot = Path(map_path).with_suffix(".png")
            if screenshot.exists():
                story.append(Paragraph("Карта маршрутов", style_normal))
                img = RLImage(str(screenshot), width=16*cm, height=12*cm)
                img.hAlign = 'CENTER'
                story.append(KeepTogether(img))

        story.append(PageBreak())

    # Заключение
    story.append(Paragraph("Заключение и рекомендации", style_h1))
    coverage = results["total_visits_planned"] / total_visits_initial if total_visits_initial else 0
    conclusion = f"""
    <font name="{DEFAULT_FONT}">
    Геопланировщик выполнил оптимизацию маршрутов с учётом бизнес-ограничений:<br/><br/>
    • Максимум 4 часа в дороге в день<br/>
    • 30 минут на обслуживание точки<br/>
    • Равномерная нагрузка по дням<br/>
    • Автоматический отсев дальних точек<br/>
     <b>Достигнутый охват: {coverage:.1%} от исходного плана</b><br/><br/>
    Рекомендации:<br/>
    • При росте точек — увеличить количество кластеров<br/>
    • Регулярно обновлять приоритеты точек<br/>
    • Для максимальной точности — использовать локальный OSRM сервер
    </font>
    """
    story.append(Paragraph(conclusion, style_normal))

    doc.build(story)
    print(f"Профессиональный PDF-отчёт сохранён: {output_path}")