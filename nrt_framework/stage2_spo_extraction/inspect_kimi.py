"""Quick DOM inspector — finds the right selectors for kimi.com chat."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

CHROME_PROFILE = Path.home() / "Library/Application Support/Google/Chrome/Default"

async def inspect():
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(CHROME_PROFILE),
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = await browser.new_page()
        await page.goto("https://www.kimi.com/", wait_until="networkidle")
        await asyncio.sleep(3)

        # Screenshot current state
        await page.screenshot(path="kimi_state.png", full_page=False)
        print("[snapshot] kimi_state.png saved")

        # Dump all input/button/textarea/contenteditable elements
        elements = await page.evaluate("""() => {
            const sel = 'input, textarea, [contenteditable], button, [role="textbox"]';
            return Array.from(document.querySelectorAll(sel)).slice(0, 30).map(el => ({
                tag: el.tagName,
                type: el.type || '',
                role: el.getAttribute('role') || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                placeholder: el.getAttribute('placeholder') || '',
                contenteditable: el.getAttribute('contenteditable') || '',
                classes: el.className.slice(0, 80),
                text: el.innerText?.slice(0, 50) || '',
                id: el.id || '',
            }));
        }""")
        print("\n=== Interactive elements ===")
        for e in elements:
            print(e)

        await asyncio.sleep(2)
        await browser.close()

asyncio.run(inspect())
