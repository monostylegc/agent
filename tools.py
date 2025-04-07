from playwright.async_api import async_playwright
import os.path
from dotenv import load_dotenv
import os

from browser_use import Agent
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig, BrowserContext

from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import PubMedRetriever
from pydantic import BaseModel

load_dotenv()

# 브라우저 초기화 및 로그인 함수
async def login(headless=False):
    """Playwright 브라우저를 초기화하고 로그인합니다."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        # storage_state.json 파일이 있는지 확인
        context = await browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            storage_state=os.path.exists("storage_state.json") and "storage_state.json" or None
        )
        
        page = await context.new_page()
    
        # 로그인 페이지로 이동하여 리다이렉트 확인
        await page.goto('https://medline.inje.ac.kr/login')
        await page.wait_for_load_state("networkidle")
    
        # 현재 URL이 로그인 페이지가 아니면 이미 로그인된 상태
        current_url = page.url
        if not current_url.endswith('/login'):
            print('이미 로그인되어 있습니다.')

        else:
            print('저장된 세션이 만료되었습니다. 다시 로그인합니다.')
            await page.goto('https://medline.inje.ac.kr/login')
            await page.wait_for_load_state("networkidle")
            await page.fill('input[name="id"]', os.getenv("MEDLINE_ID"))
            await page.fill('input[name="password"]', os.getenv("MEDLINE_PASSWORD"))
            await page.press('input[name="password"]', 'Enter')
            await page.wait_for_load_state("networkidle")
            # Playwright의 state 저장
            # await context.storage_state(path="storage_state.json")
            # 쿠키 정보 저장
            cookies = await context.cookies()
            with open("cookies.json", "w") as f:
                import json
                json.dump(cookies, f)
            print('로그인 성공. 세션 정보가 저장되었습니다.')


@tool
async def scrape_with_agent(url: str, task: str):
    """
    웹 에이전트에게 url을 제공하면 task를 수행합니다.
    task를 상세하게 알려줘야 제대로 수행할 수 있습니다.

    Args:
        url: str - 접근할 URL 
        task: str - 논문 페이지에서 수행할 작업에 대한 상세 지시사항
    
    Returns:
        str: 에이전트가 수행한 작업의 결과
    
    Examples:
        작업 예시: 
    """
    browser_config = BrowserContextConfig(
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        cookies_file="cookies.json"
    )

    browser = Browser()
    context = BrowserContext(browser=browser, config=browser_config)
    
    agent = Agent(
        browser_context=context,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        task=f"""{url}의 주소를 방문해주세요. 아래는 수행해야하는 작업입니다.
                <task>
                {task}
                </task>
              """
    )

    history = await agent.run()
    
    result = history.final_result()

    await browser.close()

    return result

@tool
def search_pubmed(query: str) -> List[Document]:
    """
    PubMed 데이터베이스에서 의학 문헌을 검색합니다.
    
    이 도구는 PubMed API를 통해 상위 10개의 검색 결과를 반환합니다.
    각 결과는 논문의 제목, URL(인제대학교 프록시 서버를 통한 접근 링크), 초록 및 저자 정보를 포함합니다.
    
    Args:
        query: str - PubMed에서 검색할 의학 용어, MeSH 용어, 저자명 등의 검색어
    
    Returns:
        List[Document]: 검색 결과를 포함하는 문서 목록. 각 문서는 다음 정보를 포함합니다:
            - title: 논문 제목
            - url: 인제대학교 프록시를 통한 원문 접근 URL
            - abstract: 논문 초록
            - authors: 저자 목록
    
    효과적인 검색 방법:
    1. MeSH 용어 활용: 
       - "heart attack" 대신 "myocardial infarction[MeSH]" 사용
       - "cancer" 대신 "neoplasms[MeSH]" 사용
    
    2. 불리언 연산자:
       - AND: 모든 검색어 포함 (예: "diabetes AND metformin")
       - OR: 어느 한 검색어 포함 (예: "covid OR sars-cov-2")
       - NOT: 특정 검색어 제외 (예: "breast cancer NOT male")
    
    3. 필드 태그:
       - [Title]: 제목에서만 검색 (예: "alzheimer[Title]")
       - [Abstract]: 초록에서만 검색 (예: "biomarker[Abstract]")
       - [Author]: 저자 검색 (예: "kim j[Author]")
       - [MeSH]: MeSH 용어 검색 (예: "hypertension[MeSH]")
    
    4. 검색 제한:
       - 출판 유형: "Randomized Controlled Trial[Publication Type]"
       - 출판 날짜: "("2020/01/01"[Date - Publication] : "2023/12/31"[Date - Publication])"
       - 언어: "English[Language]", "Korean[Language]"
    
    5. 절단 및 와일드카드:
       - *: 여러 문자 대체 (예: "therap*" → therapy, therapies, therapeutic 등)
       - ?: 한 문자 대체 (예: "wom?n" → woman, women)
    
    Examples:
        1. 기본 검색: "당뇨병 치료"
        2. MeSH 용어: "diabetes mellitus[MeSH] AND metformin[MeSH]"
        3. 필드 제한: "covid-19[Title] AND vaccine[Title]"
        4. 복합 검색: "(stroke[MeSH] OR cerebrovascular accident[Title/Abstract]) AND rehabilitation AND ("2020"[Date - Publication] : "2023"[Date - Publication])"
        5. 저자 검색: "kim j[Author] AND korea[Affiliation]"
        6. 한국 연구: "korea[Affiliation] AND hypertension[MeSH]"
        7. 체계적 문헌고찰: "systematic review[Publication Type] AND cancer screening"
    """
    retriever = PubMedRetriever(top_k_results=10)
    class PubMedResult(BaseModel):
        title: str
        url: str
        abstract: str
        authors: List[str]
    response =  retriever.invoke(query)
    result : PubMedResult = [] 

    for doc in response:
        doi = doc.metadata.get("DOI", "")
        url = f"https://doi-org-ssl.mproxy.inje.ac.kr/{doi}" if doi else ""
        
        # 메타데이터에서 필요한 정보 추출, 없는 경우 기본값 사용
        title_data = doc.metadata.get("Title", "제목 없음")
        
        # 타이틀이 딕셔너리인 경우 처리 (XML 형식으로 반환되는 경우)
        if isinstance(title_data, dict):
            if "#text" in title_data:
                title = title_data["#text"]
            else:
                # 다른 키가 있는 경우 첫 번째 텍스트 값 사용
                title = next((v for v in title_data.values() if isinstance(v, str)), "제목 없음")
        else:
            title = str(title_data)
            
        abstract = doc.metadata.get("abstract", "")
        authors = doc.metadata.get("authors", [])
        
        # 저자 리스트도 문자열 처리
        if authors and not isinstance(authors[0], str):
            try:
                authors = [str(author) for author in authors]
            except:
                authors = []
        
        result.append(PubMedResult(
            title=title,
            url=url,
            abstract=abstract,
            authors=authors,
        ))
    return result

# dummy_doi = "10.1016/j.spinee.2023.05.012"
# dummy_question = "논문의 제목은 무엇인가?"

# async def main():
#     await login()
#     result = await scrape_with_agent(dummy_doi, dummy_question)
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(main())

