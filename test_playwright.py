import asyncio
from playwright.async_api import async_playwright

async def main():
    proxy_server = 'http://31.59.20.176:6754'
    proxy_auth = {'username': 'edtilhmt', 'password': 'iblj7uuixsmy'}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(proxy={"server": proxy_server, "username": proxy_auth['username'], "password": proxy_auth['password']})
        page = await context.new_page()
        
        print("Navigating...")
        await page.goto("https://safe.trade/api/v2/trade/public/markets/prlusdt/depth", wait_until="networkidle")
        print("Page title:", await page.title())
        content = await page.content()
        print("Content starts with:", content[:100])
        
        cookies = await context.cookies()
        print("Cookies:", cookies)
        
        await browser.close()

asyncio.run(main())
