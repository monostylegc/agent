from langchain_community.retrievers import PubMedRetriever
from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any
import asyncio
from bs4 import BeautifulSoup

from browser import Browser

# 비동기 초기화를 위한 전역 변수
browser = None

async def initialize_browser():
    """Browser 객체를 비동기적으로 초기화합니다."""
    global browser
    browser = Browser()
    # 비동기 초기화 로직 실행
    browser.browser, browser.context, browser.page = await browser._login()

# 초기화 함수 실행
loop = asyncio.get_event_loop()
if not loop.is_running():
    loop.run_until_complete(initialize_browser())
else:
    # 이미 실행 중인 이벤트 루프가 있는 경우 (예: Jupyter Notebook)
    asyncio.create_task(initialize_browser())

@tool
def search_pubmed(query: str) -> List[Document]:
    """PubMed에서 논문을 검색합니다."""
    retriever = PubMedRetriever(top_k_results=5)
    return retriever.invoke(query)

@tool
async def scrape_page_with_doi(doi: str) -> Document:
    """DOI를 입력받아 해당 페이지를 스크래핑합니다."""
    global browser
    
    # browser가 초기화되지 않았다면 초기화
    if browser is None:
        await initialize_browser()
    
    title, soup = await browser.scrape_page_with_doi(doi)
    
    return Document(page_content=str(soup), metadata={"title": title, "doi": doi})

