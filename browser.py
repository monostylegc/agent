from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
import bs4
import os.path
import asyncio
from typing import List, Tuple, Dict, Any

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
            await page.fill('input[name="id"]', '108514')
            await page.fill('input[name="password"]', 'gs12341234!')
            await page.press('input[name="password"]', 'Enter')
            await page.wait_for_load_state("networkidle")
            # Playwright의 state 저장
            await context.storage_state(path="storage_state.json")
            print('로그인 성공. 세션 정보가 저장되었습니다.')

async def scrape_page_with_doi(doi: str, timeout: int = 60000) -> Tuple[str, BeautifulSoup]:
    """DOI를 사용하여 페이지를 스크랩합니다."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)        
        context = await browser.new_context(
            viewport={"width": 390, "height": 844},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            storage_state="storage_state.json"
        )

        page = await context.new_page()
        try:
            url = f"https://doi-org-ssl.mproxy.inje.ac.kr/{doi}"
            
            await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("networkidle", timeout=timeout)

            html = await page.content()
           
            soup = BeautifulSoup(html, 'html.parser')
            
            # 스크래핑 후 HTML 정리
            soup = clean_soup(soup)
            
            return soup
        except Exception as e:
            print(f"DOI {doi} 스크랩 중 오류 발생: {e}")
            # 빈 결과 대신 오류 표시와 최소 결과 반환
            return f"Error: {e}", BeautifulSoup("<html><body><p>Error occurred</p></body></html>", "html.parser")
        finally:
            await page.close()


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """HTML에서 불필요한 요소를 제거하여 정리합니다"""
    # 불필요한 태그 제거
    for tag in soup.find_all(
        [
            "script",
            "style",
            "footer",
            "header",
            "nav",
            "menu",
            "sidebar",
            "svg",
        ]
    ):
        tag.decompose()

    # 특정 클래스를 가진 태그 제거
    disallowed_class_set = {"nav", "menu", "sidebar", "footer", "advertisement", "ad", "banner"}

    # 특정 클래스를 가진 태그 확인 함수
    def does_tag_have_disallowed_class(elem) -> bool:
        if not isinstance(elem, bs4.Tag):
            return False

        return any(
            cls_name in disallowed_class_set for cls_name in elem.get("class", [])
        )

    # 특정 클래스를 가진 태그 제거
    for tag in soup.find_all(does_tag_have_disallowed_class):
        tag.decompose()
        
    # 특정 ID를 가진 태그 제거
    disallowed_id_set = {"nav", "menu", "sidebar", "footer", "header", "advertisement", "ad", "banner"}
    for tag in soup.find_all(id=lambda x: x and x.lower() in disallowed_id_set):
        tag.decompose()
        
    # 빈 태그 제거
    for tag in soup.find_all():
        if tag.name != 'br' and not tag.contents and not tag.string:
            tag.decompose()

    return soup
