"""Step-by-step debug of one extraction."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

CHROME_PROFILE = Path.home() / "Library/Application Support/Google/Chrome/Default"

async def run():
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

        # Check model
        model = await page.locator('div.model-name').inner_text()
        print(f"[model] Current: {model}")

        # Switch to Thinking if not already
        if "Thinking" not in model:
            await page.locator('div.current-model').click()
            await asyncio.sleep(0.8)
            await page.locator('text="K2.6 Thinking"').first.click()
            await asyncio.sleep(0.8)
            model = await page.locator('div.model-name').inner_text()
            print(f"[model] Switched to: {model}")

        # Send message
        box = page.locator('div.chat-input-editor')
        await box.click()
        await box.type('Output ONLY raw JSON, no markdown: [{"subj":"MoE","pred":"activates","obj":"32B params","conf":0.99,"dikw":"Knowledge"}]', delay=10)
        await page.keyboard.press("Enter")
        print("[sent] message sent, waiting...")

        # Wait for completion signal
        for attempt in range(24):  # 24 x 5s = 120s max
            await asyncio.sleep(5)
            # Check various completion signals
            actions = await page.locator('div.segment-assistant-actions').count()
            code_blocks = await page.locator('div.syntax-highlighter').count()
            markdown = await page.locator('div.markdown-container').count()
            print(f"  [{attempt*5}s] segment-assistant-actions:{actions} syntax-highlighter:{code_blocks} markdown-container:{markdown}")
            if actions > 0 or code_blocks > 0:
                print(f"  [done] Response detected at {attempt*5}s")
                break

        await page.screenshot(path="debug_response.png")

        # Try all text extraction methods
        print("\n=== Text extraction attempts ===")
        for sel in ['div.syntax-highlighter', 'div.markdown-container', 'div.markdown',
                    'pre', 'code', 'div.segment-assistant-actions ~ div',
                    'div.block-item']:
            els = page.locator(sel)
            count = await els.count()
            if count > 0:
                text = await els.nth(count-1).inner_text()
                print(f"{sel} [{count}]: {text[:150]!r}")
            else:
                print(f"{sel}: NOT FOUND")

        await browser.close()

asyncio.run(run())
