"""Find send button and response area selectors after sending a message."""
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

        # Check model switcher options
        switcher = page.locator('text="K2.6 Instant"').first
        if await switcher.count() > 0:
            print("[model] Found K2.6 Instant switcher, clicking...")
            await switcher.click()
            await asyncio.sleep(1)
            await page.screenshot(path="kimi_model_menu.png")
            print("[model] kimi_model_menu.png saved")
            # Dump menu items
            options = await page.evaluate("""() =>
                Array.from(document.querySelectorAll('[role="menuitem"], [role="option"], li')).slice(0,20).map(e => ({
                    text: e.innerText?.slice(0,60),
                    classes: e.className.slice(0,60),
                }))
            """)
            print("Menu options:", options)
            await page.keyboard.press("Escape")
            await asyncio.sleep(1)

        # Type a short message and find send button
        box = page.locator('div.chat-input-editor')
        await box.click()
        await box.type("hi", delay=50)
        await asyncio.sleep(1)

        # Find send button
        buttons = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('button, [role="button"]')).map(e => ({
                ariaLabel: e.getAttribute('aria-label') || '',
                classes: e.className.slice(0, 80),
                text: e.innerText?.slice(0, 30),
                type: e.type || '',
                disabled: e.disabled,
            }))
        """)
        print("\n=== Buttons after typing ===")
        for b in buttons:
            if b['text'] or b['ariaLabel']:
                print(b)

        await page.screenshot(path="kimi_typed.png")
        print("\nkimi_typed.png saved")
        await browser.close()

asyncio.run(inspect())
