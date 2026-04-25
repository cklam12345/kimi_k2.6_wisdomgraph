"""Send a message and capture response state + selectors after 30s."""
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
        await asyncio.sleep(2)

        # Type and send
        box = page.locator('div.chat-input-editor')
        await box.click()
        await box.type('Output ONLY this JSON array, nothing else: [{"subj":"sky","pred":"is","obj":"blue","conf":0.99,"dikw":"Knowledge"}]', delay=10)
        await page.keyboard.press("Enter")
        print("[sent] waiting 30s for response...")
        await asyncio.sleep(30)

        await page.screenshot(path="kimi_after30s.png")

        # Dump ALL div classes to find response container
        all_divs = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('div[class]'))
            .map(e => e.className.split(' ')[0])
            .filter((v, i, a) => a.indexOf(v) === i)  // unique
            .filter(c => c.length > 3)
        """)
        print("Unique div classes:", all_divs)

        # Find anything with JSON content
        matches = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('*'))
            .filter(e => e.children.length === 0 && e.innerText?.includes('"subj"'))
            .map(e => ({tag: e.tagName, classes: e.className, text: e.innerText?.slice(0,200)}))
        """)
        print("\n=== Elements containing JSON ===")
        for m in matches:
            print(m)

        await browser.close()

asyncio.run(inspect())
