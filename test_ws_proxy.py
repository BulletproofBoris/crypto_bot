import asyncio
import websockets
import os

proxy_url = os.environ.get("MY_PROXY")
print("Using proxy:", proxy_url)

async def test():
    try:
        async with websockets.connect(
            "wss://safe.trade/api/v2/websocket/public", 
            additional_headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            proxy=proxy_url,
            open_timeout=5
        ) as ws:
            print("Connected successfully!")
            return
    except Exception as e:
        print("Failed:", repr(e))

asyncio.run(test())
