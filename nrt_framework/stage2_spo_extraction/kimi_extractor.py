"""
Stage 2 — SPO Extraction via Kimi K2.6 Thinking (Playwright)
Uses existing Chrome session. Model: K2.6 Thinking (deep reasoning mode).

Run:
    python3 kimi_extractor.py --batch 20 --output ../artifacts/spo_triples
"""
import asyncio, json, argparse
from pathlib import Path
from datetime import datetime
from playwright.async_api import async_playwright

KIMI_URL  = "https://www.kimi.com/"
CHROME_PROFILE = Path.home() / "Library/Application Support/Google/Chrome/Default"

SPO_PROMPT = """Extract exactly 20 Subject-Predicate-Object triples about the topic below.
Output ONLY a raw JSON array — no markdown, no explanation, no code block.

Schema: [{{"subj":"...","pred":"...","obj":"...","conf":0.95,"dikw":"Knowledge"}}]
dikw must be one of: Data, Information, Knowledge, Wisdom

Topic: {topic}"""

PROBE_TOPICS = [
    "transformer self-attention mechanisms and computational complexity",
    "mixture of experts routing algorithms in large language models",
    "chain of thought reasoning and emergent capabilities in LLMs",
    "scaling laws for neural language models",
    "reinforcement learning from human feedback and RLHF alignment",
    "compiler optimization techniques and intermediate representations",
    "distributed training strategies for trillion parameter models",
    "memory hierarchy and cache coherence in modern processors",
    "graph neural networks and message passing algorithms",
    "dynamic programming and optimal substructure in algorithms",
    "protein folding energy minimization and AlphaFold",
    "quantum entanglement Bell inequalities and quantum computing",
    "CRISPR-Cas9 gene editing mechanisms and off-target effects",
    "neural plasticity long-term potentiation and memory formation",
    "dark matter candidates WIMPs axions and detection methods",
    "multi-agent coordination emergent swarm behavior and game theory",
    "tool use in large language models and function calling protocols",
    "retrieval augmented generation and vector similarity search",
    "knowledge graph construction from unstructured text",
    "model quantization tradeoffs accuracy compression INT4 INT8",
    "attention is all you need architecture and positional encoding",
    "low rank adaptation LoRA and parameter efficient fine tuning",
    "speculative decoding and KV cache optimization",
    "mixture of depths and adaptive compute in transformers",
    "constitutional AI and RLAIF preference learning",
    "sparse autoencoders and mechanistic interpretability",
    "in-context learning and few-shot prompting mechanisms",
    "neural scaling laws Chinchilla optimal compute allocation",
    "flash attention memory efficient transformers IO complexity",
    "mamba state space models and selective state spaces",
]


async def wait_for_response_complete(page, timeout=120):
    """Wait until Kimi finishes — segment-assistant-actions appears when done."""
    await page.wait_for_selector(
        'div.segment-assistant-actions',
        timeout=timeout * 1000,
        state="visible",
    )


async def get_last_assistant_text(page) -> str:
    """Get text from last assistant markdown-container (skip user message at index 0)."""
    blocks = page.locator('div.markdown-container')
    count = await blocks.count()
    if count < 2:
        return ""
    # index 0 = user message echo, index count-1 = last assistant response
    return await blocks.nth(count - 1).inner_text()


async def extract_triples(page, topic: str) -> list[dict]:
    prompt = SPO_PROMPT.format(topic=topic)

    box = page.locator('div.chat-input-editor')
    await box.click()
    await box.fill(prompt)      # fill() clears then sets — no keyboard shortcut issues
    await page.keyboard.press("Enter")
    print(f"  [sent] prompt typed and sent")

    try:
        await wait_for_response_complete(page, timeout=120)
        print(f"  [done] response complete")
    except Exception as e:
        print(f"  [timeout] {e} — grabbing partial")

    raw = await get_last_assistant_text(page)
    print(f"  [raw] {raw[:120]!r}")

    triples = []
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start >= 0 and end > start:
            triples = json.loads(raw[start:end])
    except json.JSONDecodeError:
        print(f"  [WARN] JSON parse failed")

    return triples


async def new_chat(page):
    """Navigate to home — most reliable way to start a fresh chat."""
    await page.goto(KIMI_URL, wait_until="networkidle")
    await page.wait_for_selector('div.chat-input-editor', timeout=10000)
    await asyncio.sleep(1)


async def run(output_dir: Path, batch: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"spo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    stats = {"topics": 0, "triples": 0, "errors": 0}

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(CHROME_PROFILE),
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = await browser.new_page()
        await page.goto(KIMI_URL, wait_until="networkidle")
        await asyncio.sleep(2)

        # Switch to K2.6 Thinking
        instant = page.locator('text="K2.6 Instant"').first
        if await instant.count() > 0:
            await instant.click()
            await asyncio.sleep(0.8)
            await page.locator('text="K2.6 Thinking"').first.click()
            await asyncio.sleep(0.8)
            print("[model] K2.6 Thinking selected")

        topics = PROBE_TOPICS[:batch]
        print(f"[NRT] Extracting {len(topics)} topics → {out_file}\n")

        with open(out_file, "a") as f:
            for i, topic in enumerate(topics):
                print(f"[{i+1}/{len(topics)}] {topic[:65]}")
                try:
                    await new_chat(page)
                    triples = await extract_triples(page, topic)
                    for t in triples:
                        t.update({"source_topic": topic,
                                  "extracted_at": datetime.now().isoformat(),
                                  "model": "kimi-k2.6-thinking"})
                        f.write(json.dumps(t) + "\n")
                    f.flush()
                    stats["triples"] += len(triples)
                    stats["topics"] += 1
                    print(f"  → {len(triples)} triples (total: {stats['triples']})")
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  [ERROR] {e}")

                await asyncio.sleep(2)

        await browser.close()

    print(f"\n[NRT] Done — topics: {stats['topics']}, "
          f"triples: {stats['triples']}, errors: {stats['errors']}")
    print(f"[NRT] Saved → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch",  type=int, default=20, help="Topics to probe")
    ap.add_argument("--output", type=Path, default=Path("../artifacts/spo_triples"))
    args = ap.parse_args()
    asyncio.run(run(args.output, args.batch))
