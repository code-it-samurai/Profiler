import logging
from typing import Optional
from bs4 import BeautifulSoup
import httpx

logger = logging.getLogger(__name__)

# Domains that require JS rendering (Playwright)
JS_REQUIRED_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
}


async def scrape_page(
    url: str,
    max_chars: int = 12000,
    wait_for_selector: Optional[str] = None,
) -> dict:
    """Scrape a web page and return structured content.

    Uses Playwright for JS-heavy sites, httpx+BS4 for static pages.

    Args:
        url: Full URL to scrape.
        max_chars: Maximum characters of text to return.
        wait_for_selector: CSS selector to wait for (Playwright only).

    Returns:
        Dict with keys: url, title, text, meta_description, success, error.
    """
    from urllib.parse import urlparse
    from profiler.tools.robots import is_scrapable  # lazy import

    # Check robots.txt before scraping — saves wasted HTTP + LLM cost
    if not await is_scrapable(url):
        logger.info(f"Skipping {url} — blocked by robots.txt")
        return {
            "url": url,
            "title": "",
            "text": "",
            "meta_description": "",
            "success": False,
            "error": "blocked_by_robots_txt",
        }

    domain = urlparse(url).netloc.replace("www.", "")

    needs_js = any(js_domain in domain for js_domain in JS_REQUIRED_DOMAINS)

    if needs_js:
        return await _scrape_with_playwright(url, max_chars, wait_for_selector)
    else:
        return await _scrape_with_httpx(url, max_chars)


async def _scrape_with_httpx(url: str, max_chars: int) -> dict:
    """Static page scraping with httpx + BeautifulSoup."""
    try:
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Profiler/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        text = soup.get_text(separator="\n", strip=True)[:max_chars]

        return {
            "url": url,
            "title": title,
            "text": text,
            "meta_description": meta_desc,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return {
            "url": url,
            "title": "",
            "text": "",
            "meta_description": "",
            "success": False,
            "error": str(e),
        }


async def _scrape_with_playwright(
    url: str, max_chars: int, wait_for: Optional[str] = None
) -> dict:
    """JS-rendered page scraping with Playwright."""
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=20000, wait_until="domcontentloaded")

            if wait_for:
                try:
                    await page.wait_for_selector(wait_for, timeout=5000)
                except Exception:
                    pass  # proceed anyway

            title = await page.title()
            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)[:max_chars]
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"]

        return {
            "url": url,
            "title": title,
            "text": text,
            "meta_description": meta_desc,
            "success": True,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Playwright scrape failed for {url}: {e}")
        return {
            "url": url,
            "title": "",
            "text": "",
            "meta_description": "",
            "success": False,
            "error": str(e),
        }
