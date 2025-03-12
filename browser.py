from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
import bs4

class Browser:
    def __init__(self):
        self.browser, self.context, self.page = self._login()

    async def _login(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(
                viewport={"width": 390, "height": 844},
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            )
            
            page = await context.new_page()
            
            await page.goto('https://medline.inje.ac.kr/login')
            await page.fill('input[name="id"]', '108514')
            await page.fill('input[name="password"]', 'gs12341234!')
            await page.press('input[name="password"]', 'Enter')
            await page.wait_for_load_state("networkidle")
        
            # Playwright의 쿠키 정보 가져오기
            # cookies = await context.cookies()
            # formatted_cookies = [
            #     {
            #         "name": cookie["name"],
            #         "value": cookie["value"],
            #         "url": f"https://{cookie['domain']}"  # domain 대신 url만 사용
            #     } for cookie in cookies
            # ]

            # Playwright의 state 저장
            await context.storage_state()
            await context.storage_state(path="storage_state.json")

            return browser, context, page

    async def scrape_page_with_doi(self, doi : str) :
        url = f"https://doi-org-ssl.mproxy.inje.ac.kr/{doi}"
        
        await self.page.goto(url)
        await self.page.wait_for_load_state("networkidle")

        html = await self.page.content()
        title = await self.page.title()
        safe_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        soup = BeautifulSoup(html, 'html.parser')
        
        # 스크래핑 후 HTML 정리
        soup = self.clean_soup(soup)

        return safe_title, soup

    def clean_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
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

    async def close(self):
        await self.browser.close()

