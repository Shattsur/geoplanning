# src/visualization/screenshot.py

from playwright.sync_api import sync_playwright
from pathlib import Path
import time

def html_to_png(html_path: Path, png_path: Path, width: int = 1400, height: int = 1000):
    """Конвертирует HTML-карту Folium в PNG-скриншот."""
    html_path = html_path.resolve()
    png_path = png_path.resolve()
    png_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": width, "height": height})
        page.goto(f"file://{html_path}")
        time.sleep(3)  # Ждём загрузки карты и тайлов
        page.screenshot(path=str(png_path), full_page=False)
        browser.close()
    print(f"Скриншот карты сохранён: {png_path}")