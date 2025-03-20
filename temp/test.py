import asyncio
import os
from temp.collect1 import search_pubmed, scrape_article, embedding_article
from langchain_core.documents import Document
from temp.temp_browser import login
from dotenv import load_dotenv

load_dotenv()

async def test_pipeline():
    """논문 검색, 스크래핑, 임베딩 파이프라인을 테스트합니다."""
    print("=" * 50)
    print("테스트 시작")
    print("로그인 중...")
    await login()
    print("로그인 완료")
    print("=" * 50)
    
    # 1. PubMed 검색
    search_query = "interbody fusion expandable cage"
    print(f"\n1. PubMed 검색 중: '{search_query}'")
    search_results = await search_pubmed(search_query)
    print(search_results)
    print(f"검색 결과: {len(search_results)}개 논문 찾음")
    
    #2. 논문 스크래핑
    print("\n2. 논문 스크래핑 중...")
    scraped_files = await scrape_article(search_results)
    print(f"스크래핑 완료: {len(scraped_files)}개 파일 추출")
    
    # 3. 임베딩
    print("\n3. 임베딩 중...")
    embedding_article(scraped_files)
    print("임베딩 완료")

    
    print("\n테스트 완료")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
