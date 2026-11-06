# /// script
# dependencies = ["playwright"]
# ///
"""Vidéo de démo de Vigie, en headless avec Playwright, serveur lancé sur le port 8740.

    uv run --with playwright playwright install chromium
    uv run scripts/record_demo.py <scan_id> <dossier_sortie>
"""
import asyncio
import sys

from playwright.async_api import async_playwright

BASE = "http://localhost:8740"

async def main(scan_id, out):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox", "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        ctx = await b.new_context(viewport={"width": 1600, "height": 900}, record_video_dir=out, record_video_size={"width": 1600, "height": 900})
        pg = await ctx.new_page()
        await pg.goto(f"{BASE}/?scan={scan_id}", wait_until="networkidle")
        await pg.wait_for_timeout(4000)
        for i in (0, 2, 5):
            await pg.locator(".vessel").nth(i).click()
            await pg.wait_for_timeout(2500)
        await pg.locator(".seg button", has_text="Rapport").click()
        await pg.wait_for_timeout(3500)
        await pg.mouse.wheel(0, 0)
        await pg.locator(".seg button", has_text="Activité").click()
        await pg.wait_for_timeout(3500)
        await ctx.close()
        await b.close()

asyncio.run(main(sys.argv[1], sys.argv[2]))
