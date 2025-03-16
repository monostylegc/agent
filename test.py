from browser import login, scrape_page_with_doi
import asyncio

async def test():
    await login()
    title, soup = await scrape_page_with_doi("10.1016/j.jclinepi.2024.01.010")
    print(title)
    print(str(soup))

asyncio.run(test())
