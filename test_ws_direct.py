import asyncio
import websockets

async def test():
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with websockets.connect(
            "wss://safe.trade/api/v2/websocket/public",
            host="104.26.12.76",
            additional_headers=headers,
            open_timeout=5
        ) as ws:
            print("Successfully connected directly via IP!")
            return
    except Exception as e:
        print("Direct IP connection failed:", repr(e))

asyncio.run(test())
