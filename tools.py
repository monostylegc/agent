from langchain_community.retrievers import PubMedRetriever
from langchain_core.tools import tool
from langchain_core.documents import Document
from typing import List
from browser import scrape_page_with_doi, login
import os
from langchain_docling.loader import DoclingLoader


def search_pubmed(query: str) -> List[Document]:
    """PubMed에서 논문을 검색합니다."""
    retriever = PubMedRetriever(top_k_results=5)
    return retriever.invoke(query)

async def scrape_article(docs: List[Document]) -> List[str]:
    """DOI를 입력받아 해당 페이지를 스크래핑합니다."""
    results = []
    
    html_folder = "html"
    if not os.path.exists(html_folder):
        os.makedirs(html_folder)
        print(f"'{html_folder}' 폴더가 생성되었습니다.")
    else:
        print(f"'{html_folder}' 폴더가 이미 존재합니다.")
    async for doc in docs:
        title = doc.metadata["title"]
        doi = doc.metadata["doi"]
        
        file_path = os.path.join(html_folder, f"{title}.html")
        if os.path.exists(file_path):
            print(f"'{file_path}' 파일이 이미 존재합니다. 스크래핑을 건너뜁니다.")
            continue
        
        content = await scrape_page_with_doi(doi)
        file_path = os.path.join(html_folder, f"{title}.html")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(content))
        results.append(file_path)
    
    return results


        