from playwright.async_api import async_playwright
import os.path
from dotenv import load_dotenv
import os

from browser_use import Agent
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig, BrowserContext

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import asyncio

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
            await context.storage_state(path="storage_state.json")
            # 쿠키 정보 저장
            cookies = await context.cookies()
            with open("cookies.json", "w") as f:
                import json
                json.dump(cookies, f)
            print('로그인 성공. 세션 정보가 저장되었습니다.')

async def scrape_with_agent(doi: str):
    browser_config = BrowserContextConfig(
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
        cookies_file="cookies.json"
    )

    browser = Browser()
    context = BrowserContext(browser=browser, config=browser_config)
    
    url = f"https://doi-org-ssl.mproxy.inje.ac.kr/{doi}"

    agent = Agent(
        browser_context=context,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        task=f"{url}을 방문해서 논문 전문을 스크랩해줘. DOI not found 라는 문장이 있으면 없는 DOI라고 출력해줘. "
    )

    result = await agent.run()
    print(result)
    await context.close()

async def main():
    await login()
    await scrape_with_agent("10.1016/j.spinee.2023.05.012")

if __name__ == "__main__":
    asyncio.run(main())
