from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rich import print

import requests
import os


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# CREATE TAVILY CLIENT
# ==================================================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==================================================
# TOOL 1: WEB SEARCH USING TAVILY
# ==================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information.
    Returns titles, URLs, and short snippets.
    """

    results = tavily.search(
        query=query,
        max_results=3
    )

    out = []

    for r in results["results"]:

        title = r.get("title", "No title")
        url = r.get("url", "No URL")
        content = r.get("content", "")

        # Limit snippet size to reduce token usage
        content = content[:400]

        out.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {content}\n"
        )

    return "\n".join(out)


# ==================================================
# TEST TAVILY SEARCH
# ==================================================

if __name__ == "__main__":

    result = web_search.invoke(
        "What are the recent news about the war?"
    )

    print("\nTAVILY SEARCH RESULT:\n")
    print(result)


# ==================================================
# TOOL 2: WEB SCRAPING USING BEAUTIFULSOUP
# ==================================================

@tool
def web_scrape(url: str) -> str:
    """
    Scrape a webpage using BeautifulSoup.
    Handles invalid, redirected, or unavailable URLs safely.
    """

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/154.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
            allow_redirects=True
        )

        # Handle HTTP errors without crashing the pipeline
        if response.status_code != 200:

            return (
                f"Unable to scrape this URL.\n"
                f"URL: {url}\n"
                f"HTTP Status: {response.status_code}\n"
                f"Please try another URL."
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # ==================================================
        # PAGE TITLE
        # ==================================================

        title = (
            soup.title.get_text(strip=True)
            if soup.title
            else "No title"
        )

        # ==================================================
        # EXTRACT PARAGRAPHS
        # ==================================================

        paragraphs = []

        for p in soup.find_all("p"):

            text = p.get_text(
                " ",
                strip=True
            )

            if text:
                paragraphs.append(text)

        # ==================================================
        # NO USEFUL CONTENT
        # ==================================================

        if not paragraphs:

            return (
                f"Title: {title}\n\n"
                f"No useful text could be extracted from:\n"
                f"{url}"
            )

        # ==================================================
        # RETURN LIMITED CONTENT
        # ==================================================

        return (
            f"Title: {title}\n\n"
            f"URL: {response.url}\n\n"
            f"Content:\n"
            + "\n".join(paragraphs[:10])
        )

    # ==================================================
    # REQUEST ERROR
    # ==================================================

    except requests.exceptions.RequestException as e:

        return (
            f"Failed to scrape the webpage.\n"
            f"URL: {url}\n"
            f"Error: {str(e)}\n"
            f"Please try another URL."
        )

    # ==================================================
    # OTHER ERROR
    # ==================================================

    except Exception as e:

        return (
            f"Unexpected scraping error.\n"
            f"URL: {url}\n"
            f"Error: {str(e)}"
        )
