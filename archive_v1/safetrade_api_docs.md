# SafeTrade API Documentation (v2)

## Authentication
Basic Auth & HmacSha256 Signed Requests.

## Websocket channels (Public)
URL: wss://safe.trade/api/v2/websocket/public
- `global.tickers` - Get list tickers by market
- `<market>.trades` - Get list trades in market
- `<market>.depth` - Get depth in market

## REST Endpoints (Public)
- `GET /trade/public/currencies`
- `GET /trade/public/currencies/{id}`
- `GET /trade/public/markets`
- `GET /trade/public/markets/{id}`
- `GET /trade/public/markets/{id}/depth` (Params: limit) -> Response: `{ "asks": [ [ 0 ] ], "bids": [ [ 0 ] ], "sequence": 0 }`
- `GET /trade/public/markets/{id}/k-line` (Params: period, time_from, time_to, page, limit) -> Response: `[ [ {} ] ]`
- `GET /trade/public/markets/{id}/trades` (Params: page, limit) -> Response: `[ { "amount": 0, "created_at": "string", "id": 0, "market": "string", "price": 0, "side": "string", "total": 0 } ]`
- `GET /trade/public/tickers`
- `GET /trade/public/tickers/{market}`
- `GET /trade/public/trading_fees`

## REST Endpoints (Private)
- `GET /trade/account/balances/spot`
- `GET /trade/account/beneficiaries`
- `GET /trade/account/deposit_address/{currency_id}`
- `GET /trade/account/deposits`
- `GET /trade/account/deposits/{txid}`
- `GET /trade/account/members/me`
- `GET /trade/account/withdraws`
- `POST /trade/account/withdraws`
- `POST /trade/account/withdraws/generate_code`
- `GET /trade/market/orders`
- `POST /trade/market/orders`
- `GET /trade/market/orders/{id}`
- `POST /trade/market/orders/{id}/cancel`
- `GET /trade/market/trades`
- `GET /trade/market/trades/{id}`
