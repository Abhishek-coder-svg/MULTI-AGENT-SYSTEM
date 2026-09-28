from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query : str) -> str:
    """Search the web for a recent and realiable information  on a topic.Return Titles ,URLs and snippets"""
  results =  tavily.serach(query=query,max_results=5)

  return results

print(web_search.invoke("what are the recent news of war?"))
    