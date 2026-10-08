import asyncio
from curl_cffi.requests import AsyncSession

async def main():
    try:
        async with AsyncSession(impersonate="chrome110") as session:
            print("Fetching REST...")
            r = await session.get("https://safe.trade/api/v2/trade/public/markets/prlusdt/depth")
            print("REST:", r.status_code, r.text[:50])

            print("Connecting WS...")
            ws = await session.ws_connect("wss://safe.trade/api/v2/websocket/public")
            print("Connected WS!")
            await ws.send(b'{"event":"subscribe","streams":["prlusdt.ob-inc"]}')
            msg = await ws.recv()
            print("WS Message:", msg[:100])
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
