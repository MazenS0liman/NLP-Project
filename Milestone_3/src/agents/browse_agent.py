import os
import asyncio
from typing import List
from playwright.async_api import async_playwright, Page, Error as PlaywrightError, TimeoutError as PlaywrightTimeout

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "llama3-70b-8192"
SYSTEM_PROMPT = """
    ### Role:
    You are a web browser agent. Your task is collect information from the web and answer questions based on that information.

    ### Instructions:
    1. You will be given a query.
    2. Also, you will be given a response from multiple web pages.
    3. You need to extract the relevant information from the response and answer the query.

    ### Constraints:
    - You can only use the information from the response to answer the query.
    - You cannot use any external knowledge or information.
    - You cannot use any external tools or libraries.

    ### Examples:
    - Query: "search who is the president of the US at 2025"
      Response: "The president of the US at 2025 is Joe Biden."
      Answer: "Joe Biden is the president of the US at 2025."
    
    - Query: "search what is the capital of France"
        Response: "The capital of France is Paris."
        Answer: "The capital of France is Paris."

    ### Input:
    {query}

    ### Search Output:
    {context}
    ### Response:
""".strip()


from groq import Groq
client = Groq(api_key=GROQ_API_KEY)

SELECTORS = [
    "li.b_algo .b_tpcn > a.tilk",
    "#b_results .b_algo h2 a",
    "#b_results .b_algo a[href]",
    "#b_results .b_algo h2 a[href]",
    "#b_results .b_algo .b_tpcn > a.tilk", 
    "#b_results .b_algo .b_tpcn > a.tilk[href]", 
]

async def find_result_links(page: Page, max_results: int = 5):
    for sel in SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=5_000)
            links = await page.locator(sel).all()
            return links[:max_results]
        except PlaywrightTimeout as e:
            print(f"⚠️  Selector {sel} timed out. No links found.")
            continue
    # nothing matched
    return []

async def find_result_links(page: Page, max_results: int = 5):
    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except PlaywrightTimeout:
        print("⚠️ networkidle timed out — proceeding anyway")

    # 1) Wait for the list container to appear
    try:
        await page.wait_for_selector("ol#b_results li.b_algo", timeout=15_000)
    except PlaywrightTimeout:
        print("❗ ol#b_results li.b_algo never showed up; dumping HTML for debug")
        html = await page.content()
        with open("debug_no_li.html", "w", encoding="utf-8") as f:
            f.write(html)
        return []

    # 2) Pull out up to max_results result items
    containers = page.locator("ol#b_results li.b_algo")
    count = min(await containers.count(), max_results)
    results = []
    for i in range(count):
        item = containers.nth(i)
        # try headline H2 link
        link = await item.locator("h2 > a").first().element_handle()
        if not link:
            # try topic card link
            link = await item.locator(".b_tpcn > a.tilk").first().element_handle()
        if not link:
            # fallback to any <a> (will catch “More info” cards too, but better than nothing)
            link = await item.locator("a[href]").first().element_handle()
        if link:
            results.append(link)
    return results

async def search_and_scrape(
    page: Page,
    query: str,
    max_results: int = 5
) -> List[dict]:
    # 1. Go to Bing and search
    await page.goto('https://www.bing.com', wait_until='domcontentloaded')
    await page.wait_for_selector('#sb_form_q', timeout=30_000)
    await page.fill(selector='#sb_form_q', value=query, timeout=15_000)
    await page.keyboard.press('Enter')
    await page.wait_for_load_state('networkidle')
    
    urls = await find_result_links(page, max_results)
    if not urls:
        print(f"⚠️  No result links found for query: {query}")
        return []

    results = []
    for url in urls:
        tab = await page.context.new_page()
        try:
            await tab.goto(url, wait_until="domcontentloaded", timeout=30_000)
            text = await tab.locator("body").inner_text()
            results.append({"url": url, "text": text})
        except Exception as e:
            results.append({"url": url, "text": f"Error loading page: {e}"})
        finally:
            await tab.close()

    return results


async def scrape_urls(
    query: str,
    max_results: int = 10
) -> List[str]:
    async with async_playwright() as p:    
        browser = await p.chromium.launch(args=['--lang=en-US,en'])
        context = await browser.new_context(locale='en-US')
        page = await context.new_page()

        # Navigate to Bing and search
        await page.goto('https://www.bing.com', wait_until='domcontentloaded')
        await page.wait_for_selector('#sb_form_q', timeout=30_000)
        await page.fill(selector='#sb_form_q', value=query, timeout=15_000)
        await page.keyboard.press('Enter')

        # Wait for search results
        try:
            # this selector matches the result links in most Bing variants
            await page.wait_for_selector('#b_results .b_algo h2 a', timeout=15_000)
            links = await page.locator('#b_results .b_algo h2 a').all()
        except PlaywrightTimeout:
            # nothing matched – bail out with an empty list
            print("⚠️  Bing results selector timed out. No links found. for query:", query)
            return []

        urls = []
        for link in links[:max_results]:
            href = await link.get_attribute('href')
            if href:
                urls.append(href)
        
        return urls

async def get_information(query: str) -> str:
    async with async_playwright() as p:    
        browser = await p.chromium.launch(args=['--lang=en-US,en'])
        context = await browser.new_context(locale='en-US')
        page = await context.new_page()

        # If the query is a direct "search ..." command, bypass LLM
        if query.strip().lower().startswith('search'):
            # e.g. "search who is the president of the US at 2025"
            return await search_and_scrape(page, query[len('search '):].strip())

        # Otherwise, use the LLM-driven browser agent
        results = await search_and_scrape(page, query.strip())
        scrape = []

        # For each result, navigate to the URL and scrape the page
        # If the result is an error, skip it
        for result in results:
            url = result['url']
            text = result['text']
            if 'Error' in text:
                continue

            # Navigate to the URL and scrape the page
            await page.goto(url, wait_until='domcontentloaded', timeout=30_000)
            await page.wait_for_timeout(2_000)
            await page.evaluate('window.scrollTo(0, 0)')  # Scroll to top

            # Scrape
            body_handle = await page.query_selector('body')
            page_text = await body_handle.inner_text() if body_handle else ''
            scrape.append({'url': url, 'text': page_text})

        return scrape


async def search_web(query: str, use_llm: bool = False) -> str:
    # 1. Scrape the top results
    # try:
    results = await get_information(query)
    # except Exception as e:
        # return f"I do not know the answer to {query}."

    # 2. Build one big context string
    #    — we’ll truncate each page to something reasonable (e.g. 2000 chars)
    snippets = []
    for r in results:
        if 'Error' in r['text']:
            continue
        snippet = r['text'].strip().replace('\n', ' ')[:2000]
        snippets.append(f"URL: {r['url']}\nContent: {snippet}")

    context = "\n\n---\n\n".join(snippets)

    if use_llm:
        # 3. Compose the messages for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"""Here are the web search results I found for your query:

            {context}

            Please use *only* the information above to answer the user’s question below."""},
                    {"role": "user", "content": query}
        ]

        # 4. Call the LLM
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        return resp.choices[0].message.content
    else:
        return context

async def main():
    result = await get_information(query='who is the president of the US at 2000')
    print("Response:", result)

if __name__ == "__main__":
    asyncio.run(main())
