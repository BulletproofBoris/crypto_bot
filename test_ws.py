import asyncio
import websockets
async def test():
    try:
        async with websockets.connect("wss://safe.trade/api/v2/websocket/public", 
                                      user_agent_header="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                      origin="https://safe.trade",
                                      open_timeout=5) as ws:
            print("Connected to safe.trade")
            return
    except Exception as e:
        print("safe.trade failed:", e)

    try:
        async with websockets.connect("wss://safetrade.com/api/v2/websocket/public",
                                      user_agent_header="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                                      origin="https://safetrade.com",
                                      open_timeout=5) as ws:
            print("Connected to safetrade.com")
            return
    except Exception as e:
        print("safetrade.com failed:", e)

asyncio.run(test())
