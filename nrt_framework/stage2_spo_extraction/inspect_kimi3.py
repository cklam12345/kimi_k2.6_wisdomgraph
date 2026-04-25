"""Send a real message and capture response selectors."""
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

        # Switch to K2.6 Thinking
        await page.locator('text="K2.6 Instant"').first.click()
        await asyncio.sleep(1)
        await page.locator('text="K2.6 Thinking"').first.click()
        await asyncio.sleep(1)
        await page.screenshot(path="kimi_thinking_selected.png")
        print("[model] Switched to K2.6 Thinking")

        # Type and send a short message
        box = page.locator('div.chat-input-editor')
        await box.click()
        await box.type('Reply with only this JSON: [{"subj":"test","pred":"is","obj":"working"}]', delay=20)
        await asyncio.sleep(1)

        # Find and click send button (the arrow button, enabled when text present)
        send_btn = page.locator('button[type="submit"], button:has(svg):last-of-type').last
        # Try pressing Enter instead as fallback
        await page.keyboard.press("Enter")
        print("[send] Message sent via Enter")

        await asyncio.sleep(3)
        await page.screenshot(path="kimi_waiting.png")

        # Wait up to 60s for response to appear
        print("[wait] Waiting for response...")
        await asyncio.sleep(15)
        await page.screenshot(path="kimi_response.png")

        # Dump all text-containing divs to find response selector
        elements = await page.evaluate("""() => {
            const candidates = document.querySelectorAll('div, p, section');
            return Array.from(candidates)
                .filter(e => e.innerText?.length > 10 && e.innerText?.length < 500)
                .slice(0, 40)
                .map(e => ({
                    tag: e.tagName,
                    classes: e.className.slice(0, 80),
                    text: e.innerText?.slice(0, 100),
                    dataAttrs: Object.keys(e.dataset).join(','),
                }));
        }""")
        print("\n=== Response area candidates ===")
        for e in elements:
            if 'working' in (e['text'] or '') or 'subj' in (e['text'] or '') or 'test' in (e['text'] or ''):
                print("*** MATCH ***", e)
            else:
                print(e)

        await browser.close()

asyncio.run(inspect())
