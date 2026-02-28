"""
Tradovate API Client — Futures Trade Execution Engine
======================================================
Handles the full TradingView → Tradovate execution pipeline for MNQ futures.

Flow:
    1. TradingView alert fires (price action or strategy signal).
    2. Alert POSTs a JSON webhook payload to your FastAPI/Flask endpoint.
    3. TradovateWebhookProcessor.process() validates and parses the payload.
    4. It calls TradovateClient to place the appropriate order on Tradovate.
    5. Execution result (fill price, order ID, status) is returned and logged.

Authentication lifecycle:
    - On startup, call await client.authenticate() to get an accessToken.
    - Tokens expire (~24h). Call await client.renew_token() before expiry,
      or rely on the auto-renewal guard built into every request method.
    - All authenticated requests send: Authorization: Bearer <accessToken>

CME Compliance:
    - All automated orders MUST include "isAutomated": true per CME Rule 576.
      This is enforced unconditionally in every order method.

Demo vs. Live:
    - Set demo_mode=True (default) to target demo.tradovateapi.com.
    - Set demo_mode=False to target live.tradovateapi.com.
    - Market data always routes to md.tradovateapi.com regardless of mode.

Dependencies:
    aiohttp>=3.9.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError as e:
    raise ImportError(
        "aiohttp is required for TradovateClient. "
        "Install it with:  pip install aiohttp>=3.9.0"
    ) from e

logger = logging.getLogger("tradovate_client")

_LIVE_URL  = "https://live.tradovateapi.com/v1/"
_DEMO_URL  = "https://demo.tradovateapi.com/v1/"
_MD_URL    = "https://md.tradovateapi.com/v1/"

_TOKEN_RENEWAL_BUFFER_SECS = 300
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_SECS = 1.5

_TV_ACTION_MAP: Dict[str, str] = {
    "buy":   "Buy",  "sell":  "Sell",
    "long":  "Buy",  "short": "Sell",
    "Buy":   "Buy",  "Sell":  "Sell",
    "BUY":   "Buy",  "SELL":  "Sell",
    "LONG":  "Buy",  "SHORT": "Sell",
}

_REQUIRED_WEBHOOK_FIELDS = {"action", "symbol", "qty"}


class TradovateAuthError(Exception):
    """Raised when authentication or token renewal fails."""

class TradovateAPIError(Exception):
    """Raised when the Tradovate API returns a non-success response."""

class TradovateRateLimitError(TradovateAPIError):
    """Raised when the API returns HTTP 429 and all retries are exhausted."""

class WebhookValidationError(ValueError):
    """Raised when an incoming webhook payload fails validation."""


class TradovateClient:
    """
    Async Tradovate REST API client for MNQ futures order execution.

    Parameters
    ----------
    username : str
        Tradovate account username.
    password : str
        Tradovate account password.
    app_id : str
        Registered app name (e.g. "MNQ Hybrid Trader").
    cid : int
        Client ID issued by Tradovate during app registration.
    device_id : str
        Stable UUID that identifies this device. Generate once and persist.
    secret : str
        Client secret issued by Tradovate during app registration.
    demo_mode : bool
        If True, all requests target the demo environment (default: True).
    """

    def __init__(self, username, password, app_id, cid, device_id, secret, demo_mode=True):
        self.username   = username
        self.password   = password
        self.app_id     = app_id
        self.cid        = cid
        self.device_id  = device_id
        self.secret     = secret
        self.demo_mode  = demo_mode
        self._access_token:    Optional[str]      = None
        self._md_access_token: Optional[str]      = None
        self._token_expiry:    Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("TradovateClient init [env=%s, user=%s]", "DEMO" if demo_mode else "LIVE", username)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    def _get_base_url(self) -> str:
        return _DEMO_URL if self.demo_mode else _LIVE_URL

    def _url(self, path: str) -> str:
        return self._get_base_url().rstrip("/") + "/" + path.lstrip("/")

    def _md_url(self, path: str) -> str:
        return _MD_URL.rstrip("/") + "/" + path.lstrip("/")

    def _auth_headers(self) -> Dict[str, str]:
        if not self._access_token:
            raise TradovateAuthError("No access token — call authenticate() first.")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._access_token}"}

    def is_token_valid(self) -> bool:
        if not self._access_token or self._token_expiry is None:
            return False
        return datetime.now(tz=timezone.utc) < (self._token_expiry - timedelta(seconds=_TOKEN_RENEWAL_BUFFER_SECS))

    async def authenticate(self) -> bool:
        """Obtain a new access token from Tradovate."""
        payload = {
            "name": self.username, "password": self.password,
            "appId": self.app_id, "appVersion": "1.0",
            "cid": self.cid, "deviceId": self.device_id, "sec": self.secret,
        }
        try:
            session = await self._get_session()
            async with session.post(self._url("auth/accesstokenrequest"), json=payload) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200 or "errorText" in data:
                    logger.error("Auth failed: %s", data.get("errorText", f"HTTP {resp.status}"))
                    return False
                self._access_token    = data.get("accessToken")
                self._md_access_token = data.get("mdAccessToken")
                expiry_str = data.get("expirationTime")
                self._token_expiry = (
                    datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    if expiry_str else
                    datetime.now(tz=timezone.utc) + timedelta(hours=23)
                )
                logger.info("Auth OK. Token expires %s", self._token_expiry.isoformat())
                return True
        except aiohttp.ClientError as exc:
            logger.exception("Network error during auth: %s", exc)
            return False

    async def renew_token(self) -> bool:
        """Renew the current access token."""
        if not self._access_token:
            return await self.authenticate()
        try:
            session = await self._get_session()
            async with session.get(self._url("auth/renewAccessToken"), headers=self._auth_headers()) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200 or "errorText" in data:
                    return await self.authenticate()
                if new_token := data.get("accessToken"):
                    self._access_token = new_token
                expiry_str = data.get("expirationTime")
                self._token_expiry = (
                    datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                    if expiry_str else
                    datetime.now(tz=timezone.utc) + timedelta(hours=23)
                )
                return True
        except aiohttp.ClientError:
            return await self.authenticate()

    async def _ensure_token(self) -> None:
        if not self.is_token_valid():
            if not await self.renew_token():
                raise TradovateAuthError("Unable to obtain valid access token.")

    async def _request(self, method, url, *, payload=None, use_auth=True, retry_count=0):
        headers = self._auth_headers() if use_auth else {"Content-Type": "application/json"}
        session = await self._get_session()
        try:
            kwargs: Dict[str, Any] = {"headers": headers}
            if payload is not None:
                kwargs["json"] = payload
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 429:
                    if retry_count >= _MAX_RETRIES:
                        raise TradovateRateLimitError(f"Rate limit after {_MAX_RETRIES} retries")
                    await asyncio.sleep(_RETRY_BACKOFF_BASE_SECS * (2 ** retry_count))
                    return await self._request(method, url, payload=payload, use_auth=use_auth, retry_count=retry_count+1)
                if resp.status >= 500 and retry_count < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF_BASE_SECS * (2 ** retry_count))
                    return await self._request(method, url, payload=payload, use_auth=use_auth, retry_count=retry_count+1)
                data = await resp.json(content_type=None)
                if resp.status not in (200, 201):
                    err = data.get("errorText") or data.get("error") or str(data)
                    raise TradovateAPIError(f"[HTTP {resp.status}] {method} {url} → {err}")
                if isinstance(data, dict) and "errorText" in data:
                    raise TradovateAPIError(f"Tradovate API error: {data['errorText']}")
                return data
        except (TradovateRateLimitError, TradovateAPIError, TradovateAuthError):
            raise
        except aiohttp.ClientError as exc:
            raise TradovateAPIError(f"Network error: {exc}") from exc

    async def _get(self, path):
        await self._ensure_token()
        return await self._request("GET", self._url(path))

    async def _post(self, path, payload):
        await self._ensure_token()
        return await self._request("POST", self._url(path), payload=payload)

    async def get_accounts(self) -> List[Dict[str, Any]]:
        result = await self._get("account/list")
        return result if isinstance(result, list) else result.get("items", [])

    async def get_account_balance(self, account_id: int) -> Dict[str, Any]:
        return await self._post("account/cashBalanceSnapshot", {"accountId": account_id})

    async def get_positions(self, account_id: int) -> List[Dict[str, Any]]:
        result = await self._get("position/list")
        positions = result if isinstance(result, list) else result.get("items", [])
        return [p for p in positions if p.get("accountId") == account_id and p.get("netPos", 0) != 0]

    async def liquidate_position(self, account_id: int, contract_id: int) -> Dict[str, Any]:
        return await self._post("order/liquidateposition", {
            "accountId": account_id, "contractId": contract_id,
            "admin": False, "isAutomated": True,
        })

    async def place_market_order(self, account_id, account_spec, symbol, action, qty) -> Dict[str, Any]:
        action = self._validate_action(action)
        qty    = self._validate_qty(qty)
        payload = {
            "accountSpec": account_spec, "accountId": account_id,
            "action": action, "symbol": symbol,
            "orderQty": qty, "orderType": "Market", "isAutomated": True,
        }
        return await self._post("order/placeorder", payload)

    async def place_bracket_order(self, account_id, account_spec, symbol, action, qty,
                                   take_profit_price, stop_loss_price) -> Dict[str, Any]:
        action = self._validate_action(action)
        qty    = self._validate_qty(qty)
        exit_action = "Sell" if action == "Buy" else "Buy"
        if take_profit_price <= 0 or stop_loss_price <= 0:
            raise ValueError("take_profit_price and stop_loss_price must be > 0")
        entry = {"accountSpec": account_spec, "accountId": account_id, "action": action,
                 "symbol": symbol, "orderQty": qty, "orderType": "Market", "isAutomated": True}
        bracket1 = {"action": exit_action, "symbol": symbol, "orderQty": qty,
                    "orderType": "Limit", "price": take_profit_price, "isAutomated": True}
        bracket2 = {"action": exit_action, "symbol": symbol, "orderQty": qty,
                    "orderType": "Stop", "stopPrice": stop_loss_price, "isAutomated": True}
        return await self._post("order/placeOSO", {"entry": entry, "bracket1": bracket1, "bracket2": bracket2})

    async def place_oco_order(self, account_id, account_spec, symbol, action, qty,
                              take_profit_price, stop_loss_price) -> Dict[str, Any]:
        action = self._validate_action(action)
        qty    = self._validate_qty(qty)
        tp = {"accountSpec": account_spec, "accountId": account_id, "action": action,
              "symbol": symbol, "orderQty": qty, "orderType": "Limit",
              "price": take_profit_price, "isAutomated": True}
        sl = {"accountSpec": account_spec, "accountId": account_id, "action": action,
              "symbol": symbol, "orderQty": qty, "orderType": "Stop",
              "stopPrice": stop_loss_price, "isAutomated": True}
        return await self._post("order/placeoco", {"order1": tp, "order2": sl})

    @staticmethod
    def _validate_action(action: str) -> str:
        mapped = _TV_ACTION_MAP.get(action)
        if mapped is None:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {list(_TV_ACTION_MAP.keys())}")
        return mapped

    @staticmethod
    def _validate_qty(qty: int) -> int:
        qty = int(qty)
        if qty < 1:
            raise ValueError(f"Quantity must be >= 1, got {qty}")
        return qty


class TradovateClientSync:
    """
    Synchronous wrapper around TradovateClient for use in Streamlit and other sync contexts.
    """

    def __init__(self, *args, **kwargs):
        self._async_client = TradovateClient(*args, **kwargs)

    def close(self):
        asyncio.run(self._async_client.close())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def is_token_valid(self) -> bool:
        return self._async_client.is_token_valid()

    def authenticate(self) -> bool:
        return asyncio.run(self._async_client.authenticate())

    def renew_token(self) -> bool:
        return asyncio.run(self._async_client.renew_token())

    def get_accounts(self):
        return asyncio.run(self._async_client.get_accounts())

    def get_account_balance(self, account_id):
        return asyncio.run(self._async_client.get_account_balance(account_id))

    def get_positions(self, account_id):
        return asyncio.run(self._async_client.get_positions(account_id))

    def liquidate_position(self, account_id, contract_id):
        return asyncio.run(self._async_client.liquidate_position(account_id, contract_id))

    def place_market_order(self, account_id, account_spec, symbol, action, qty):
        return asyncio.run(self._async_client.place_market_order(account_id, account_spec, symbol, action, qty))

    def place_bracket_order(self, account_id, account_spec, symbol, action, qty,
                            take_profit_price, stop_loss_price):
        return asyncio.run(self._async_client.place_bracket_order(
            account_id, account_spec, symbol, action, qty, take_profit_price, stop_loss_price))

    def place_oco_order(self, account_id, account_spec, symbol, action, qty,
                        take_profit_price, stop_loss_price):
        return asyncio.run(self._async_client.place_oco_order(
            account_id, account_spec, symbol, action, qty, take_profit_price, stop_loss_price))


class TradovateWebhookProcessor:
    """
    Parses, validates, and executes TradingView alert webhooks on Tradovate.
    """

    def __init__(self, client: TradovateClient, account_id: int, account_spec: str):
        self.client       = client
        self.account_id   = account_id
        self.account_spec = account_spec

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a TradingView webhook payload and execute the corresponding Tradovate order."""
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        self._validate(payload)

        raw_action  = str(payload["action"]).strip()
        symbol      = str(payload["symbol"]).strip().upper()
        qty         = int(payload["qty"])
        take_profit = payload.get("take_profit")
        stop_loss   = payload.get("stop_loss")
        comment     = payload.get("comment", "")

        if raw_action.lower() in ("flat", "flatten", "close", "exit", "liquidate"):
            return await self._handle_flatten(symbol, timestamp, comment)

        action = TradovateClient._validate_action(raw_action)

        try:
            if take_profit is not None and stop_loss is not None:
                result = await self.client.place_bracket_order(
                    self.account_id, self.account_spec, symbol, action, qty,
                    float(take_profit), float(stop_loss))
                order_type = "bracket"
            else:
                result = await self.client.place_market_order(
                    self.account_id, self.account_spec, symbol, action, qty)
                order_type = "market"
        except (TradovateAPIError, TradovateAuthError, ValueError) as exc:
            return {"success": False, "action": action, "symbol": symbol, "qty": qty,
                    "order_type": "market", "order_id": None, "raw": {},
                    "error": str(exc), "timestamp": timestamp, "comment": comment}

        return {"success": True, "action": action, "symbol": symbol, "qty": qty,
                "order_type": order_type, "order_id": self._extract_order_id(result),
                "raw": result, "error": None, "timestamp": timestamp, "comment": comment}

    async def _handle_flatten(self, symbol, timestamp, comment):
        try:
            positions = await self.client.get_positions(self.account_id)
        except TradovateAPIError as exc:
            return {"success": False, "action": "flat", "symbol": symbol, "qty": 0,
                    "order_type": "flat", "order_id": None, "raw": {},
                    "error": str(exc), "timestamp": timestamp, "comment": comment}
        results = []
        for pos in positions:
            try:
                r = await self.client.liquidate_position(self.account_id, pos["contractId"])
                results.append(r)
            except TradovateAPIError:
                pass
        return {"success": True, "action": "flat", "symbol": symbol,
                "qty": sum(abs(p.get("netPos", 0)) for p in positions),
                "order_type": "flat", "order_id": None,
                "raw": {"liquidations": results}, "error": None,
                "timestamp": timestamp, "comment": comment}

    @staticmethod
    def _validate(payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise WebhookValidationError(f"Payload must be a JSON object, got {type(payload).__name__}")
        missing = _REQUIRED_WEBHOOK_FIELDS - set(payload.keys())
        if missing:
            raise WebhookValidationError(f"Missing required fields: {sorted(missing)}")
        try:
            qty = int(payload["qty"])
        except (TypeError, ValueError):
            raise WebhookValidationError(f"'qty' must be integer, got: {payload['qty']!r}")
        if qty < 1:
            raise WebhookValidationError(f"'qty' must be >= 1, got {qty}")
        action = str(payload["action"]).strip().lower()
        flatten_actions = {"flat", "flatten", "close", "exit", "liquidate"}
        all_valid = {k.lower() for k in _TV_ACTION_MAP} | flatten_actions
        if action not in all_valid:
            raise WebhookValidationError(f"'action' must be one of {sorted(all_valid)}, got: {payload['action']!r}")
        for price_field in ("take_profit", "stop_loss"):
            if price_field in payload:
                try:
                    val = float(payload[price_field])
                except (TypeError, ValueError):
                    raise WebhookValidationError(f"'{price_field}' must be a number")
                if val <= 0:
                    raise WebhookValidationError(f"'{price_field}' must be > 0")

    @staticmethod
    def _extract_order_id(response: Dict[str, Any]) -> Optional[int]:
        if isinstance(response, dict):
            if "orderId" in response: return response["orderId"]
            if "id" in response: return response["id"]
            order = response.get("order")
            if isinstance(order, dict):
                return order.get("id") or order.get("orderId")
        return None


def create_client_from_env() -> TradovateClientSync:
    """
    Build a TradovateClientSync from environment variables.

    Required env vars: TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_SECRET
    Optional: TRADOVATE_APP_ID, TRADOVATE_CID, TRADOVATE_DEVICE_ID, DEMO_MODE
    """
    import os
    username = os.getenv("TRADOVATE_USERNAME", "")
    password = os.getenv("TRADOVATE_PASSWORD", "")
    secret   = os.getenv("TRADOVATE_SECRET",   "")
    if not username or not password or not secret:
        raise EnvironmentError(
            "TRADOVATE_USERNAME, TRADOVATE_PASSWORD, and TRADOVATE_SECRET must all be set."
        )
    return TradovateClientSync(
        username  = username,
        password  = password,
        app_id    = os.getenv("TRADOVATE_APP_ID",   "MNQ Hybrid Trader"),
        cid       = int(os.getenv("TRADOVATE_CID",  "8")),
        device_id = os.getenv("TRADOVATE_DEVICE_ID") or str(uuid.uuid4()),
        secret    = secret,
        demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true",
    )
