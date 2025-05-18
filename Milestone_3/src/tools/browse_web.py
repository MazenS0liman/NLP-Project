import os
import requests
from typing import List, Optional
from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# ——————————————————————————————————————————————————————————————
# This is a simple class to add colors to the output in the terminal.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'

# ——————————————————————————————————————————————————————————————
def get_search_urls(query: str, num: int = 10) -> List[str]:
    """
    Perform a Google Custom Search and return the top `num` result URLs.
    
    Args:
        query: The search query string.
        num:   How many results to request (max 10 per API limits).
    
    Returns:
        A list of result URLs (may be shorter if fewer results are found).
    """
    endpoint = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': GOOGLE_API_KEY,
        'cx':  SEARCH_ENGINE_ID,
        'q':   query,
        'num': num,
    }
    
    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()  # will raise HTTPError on bad status
    
    data = resp.json()
    items = data.get('items', [])
    
    # Extract the 'link' field from each item
    urls = [item['link'] for item in items if 'link' in item]
    return urls

def fetch_body_text(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetches only the <body> paragraph text from a webpage.
    Strips out headers, footers, navs, asides, scripts, styles, etc.
    Returns a newline-joined string of all <p> contents.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Bot/1.0; "
            "+https://example.com/bot)"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(bcolors.RED + "Error" + bcolors.ENDC + f" fetching {url}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # 1) Narrow to the <body> (or fallback to whole document)
    body = soup.body or soup

    # 2) Remove non-content tags
    for tag_name in ("script", "style", "noscript",
                     "header", "footer", "nav", "aside"):
        for tag in body.find_all(tag_name):
            tag.decompose()

    # 3) If the page has an <article>, prefer that
    container = body.find("article") or body

    # 4) Gather all <p> text
    paragraphs = [
        p.get_text(strip=True)
        for p in container.find_all("p")
        if p.get_text(strip=True)
    ]

    return "\n\n".join(paragraphs) if paragraphs else None

def fetch_text(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetches and returns the visible text content of any webpage.

    Args:
        url:     The full URL to scrape.
        timeout: Seconds to wait for the HTTP response.

    Returns:
        A single string containing the page’s visible text, or None on error.
    """
    headers = {
        # Identify yourself; some sites block default Python UA strings
        "User-Agent": "Mozilla/5.0 (compatible; Bot/1.0; +https://example.com/bot)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(bcolors.RED + "Error" + bcolors.ENDC + f" fetching {url}: {e}")
        return None

    # Parse the HTML
    soup = BeautifulSoup(resp.content, "html.parser")

    # Remove non-visible elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.extract()

    # Get the visible text
    text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def browse_web(query: str, num: int = 10) -> List[str]:
    """
    Fetches the top `num` URLs from a Google search for the given query.
    
    Args:
        query: The search query string.
        num:   How many results to request (max 10 per API limits).
    
    Returns:
        Available text content from the top `num` URLs.
    """
    try:
        urls = get_search_urls(query, num)
        print(bcolors.OKGREEN + "Found" + bcolors.ENDC + f" {len(urls)} URLs for query: {query}")

        context = []
        for url in urls:
            text = fetch_body_text(url)
            if text and "Error" not in text:
                context.append(text)

        return context
    except Exception as e:
        print(bcolors.FAIL + "Error" + bcolors.ENDC + f" during web browsing: {e}")
        return []

if __name__ == "__main__":
    search_query = "search who is the president of the US at 2025"
    content = browse_web(search_query, num=5)

    print(f"\n---\nText for: {search_query}\n")
    if content:
        print(content[:2000], "…")
    else:
        print("Failed to retrieve content.")
        