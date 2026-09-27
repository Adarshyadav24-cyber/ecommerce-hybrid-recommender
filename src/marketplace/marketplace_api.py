"""Authorized marketplace adapters for ShopSense AI.

This module never invents live prices. It returns API data only when the
corresponding credentials are configured. Amazon uses the current Creators API;
Flipkart uses its Affiliate API.
"""
import json
import os
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def _http_json(url: str, method: str = "GET", headers: Optional[dict] = None,
               payload: Optional[dict] = None, timeout: int = 20) -> dict:
    body = None
    request_headers = headers.copy() if headers else {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw[:800]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def _first(obj: Any, keys: tuple) -> Any:
    """Find the first matching key recursively in a JSON object."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, "", []):
                return obj[key]
        for value in obj.values():
            found = _first(value, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _first(value, keys)
            if found not in (None, "", []):
                return found
    return None


def _money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("amount", "value", "price"):
            if key in value:
                return _money(value[key])
    try:
        cleaned = str(value).replace(",", "")
        import re
        match = re.search(r"\d+(?:\.\d+)?", cleaned)
        return float(match.group()) if match else None
    except Exception:
        return None


def _format_price(value: Optional[float]) -> str:
    if value is None:
        return "Not available"
    return f"₹{value:,.2f}"


def amazon_configured() -> bool:
    return all([
        _secret("AMAZON_CREDENTIAL_ID"),
        _secret("AMAZON_CREDENTIAL_SECRET"),
        _secret("AMAZON_PARTNER_TAG"),
    ])


def _amazon_token() -> str:
    credential_id = _secret("AMAZON_CREDENTIAL_ID")
    credential_secret = _secret("AMAZON_CREDENTIAL_SECRET")
    # India is in Amazon's EU credential region according to current docs.
    token_url = os.getenv(
        "AMAZON_TOKEN_URL",
        "https://api.amazon.co.uk/auth/o2/token"
    )
    payload = {
        "grant_type": "client_credentials",
        "client_id": credential_id,
        "client_secret": credential_secret,
        "scope": "creatorsapi::default",
    }
    data = _http_json(token_url, method="POST", payload=payload)
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Amazon did not return an access token.")
    return token


def search_amazon(query: str) -> Dict[str, Any]:
    if not amazon_configured():
        return {
            "marketplace": "Amazon",
            "configured": False,
            "title": "Amazon API not configured",
            "price_display": "—",
            "availability": "Not connected",
            "offer": "—",
            "url": "",
            "message": (
                "Add AMAZON_CREDENTIAL_ID, AMAZON_CREDENTIAL_SECRET and "
                "AMAZON_PARTNER_TAG to your environment/Streamlit secrets."
            ),
        }

    token = _amazon_token()
    partner_tag = _secret("AMAZON_PARTNER_TAG")
    marketplace = _secret("AMAZON_MARKETPLACE", "www.amazon.in")

    payload = {
        "keywords": query,
        "partnerTag": partner_tag,
        "marketplace": marketplace,
        "resources": [
            "images.primary.large",
            "itemInfo.title",
            "itemInfo.byLineInfo",
            "offersV2.listings.price",
            "offersV2.listings.availability",
            "offersV2.listings.merchantInfo.name",
        ],
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-marketplace": marketplace,
    }
    data = _http_json(
        "https://creatorsapi.amazon/catalog/v1/searchItems",
        method="POST",
        headers=headers,
        payload=payload,
    )

    items = data.get("searchResult", {}).get("items", [])
    if not items:
        return {
            "marketplace": "Amazon",
            "configured": True,
            "title": "No Amazon result",
            "price_display": "Not available",
            "availability": "Not found",
            "offer": "—",
            "url": "",
            "message": "No product matched this query.",
        }

    item = items[0]
    title = _first(item, ("displayValue", "label")) or _first(item, ("title",)) or "Amazon product"
    price = _first(item, ("price", "currentPrice", "money"))
    amount = _money(price)
    availability = _first(item, ("displayValue",)) or _first(item, ("availability",)) or "Unknown"
    asin = _first(item, ("asin", "ASIN"))
    url = _first(item, ("detailPageURL", "detailPageUrl"))
    if not url and asin:
        url = f"https://www.amazon.in/dp/{asin}?tag={urllib.parse.quote(partner_tag)}"
    offer = _first(item, ("savingPercentage", "savings", "dealDetails"))

    return {
        "marketplace": "Amazon",
        "configured": True,
        "title": str(title),
        "price": amount,
        "price_display": _format_price(amount),
        "availability": str(availability),
        "offer": str(offer) if offer else "—",
        "url": str(url or ""),
        "message": "Data returned by Amazon Creators API.",
    }


def flipkart_configured() -> bool:
    return all([
        _secret("FLIPKART_AFFILIATE_ID"),
        _secret("FLIPKART_AFFILIATE_TOKEN"),
    ])


def search_flipkart(query: str) -> Dict[str, Any]:
    if not flipkart_configured():
        return {
            "marketplace": "Flipkart",
            "configured": False,
            "title": "Flipkart API not configured",
            "price_display": "—",
            "availability": "Not connected",
            "offer": "—",
            "url": "",
            "message": (
                "Add FLIPKART_AFFILIATE_ID and FLIPKART_AFFILIATE_TOKEN "
                "to your environment/Streamlit secrets."
            ),
        }

    params = urllib.parse.urlencode({"query": query, "resultCount": 10})
    url = f"https://affiliate-api.flipkart.net/affiliate/1.0/search/json?{params}"
    headers = {
        "Fk-Affiliate-Id": _secret("FLIPKART_AFFILIATE_ID"),
        "Fk-Affiliate-Token": _secret("FLIPKART_AFFILIATE_TOKEN"),
    }
    data = _http_json(url, headers=headers)
    products = data.get("productInfoList", [])
    if not products:
        return {
            "marketplace": "Flipkart",
            "configured": True,
            "title": "No Flipkart result",
            "price_display": "Not available",
            "availability": "Not found",
            "offer": "—",
            "url": "",
            "message": "No product matched this query.",
        }

    product = products[0]
    title = _first(product, ("title",)) or "Flipkart product"
    price = _first(product, ("sellingPrice", "selling_price", "specialPrice", "price"))
    amount = _money(price)
    availability = _first(product, ("inStock", "isAvailable", "availability"))
    if isinstance(availability, bool):
        availability = "In stock" if availability else "Out of stock"
    availability = str(availability or "Unknown")
    offer = _first(product, ("offers", "offer", "cashBack", "discount"))
    url = _first(product, ("productUrl", "productURL", "url")) or ""

    # Some feed/search responses expose price fields under nested objects.
    if amount is None:
        amount = _money(_first(product, ("sellingPrice", "price", "specialPrice")))

    return {
        "marketplace": "Flipkart",
        "configured": True,
        "title": str(title),
        "price": amount,
        "price_display": _format_price(amount),
        "availability": availability,
        "offer": str(offer) if offer not in (None, "") else "—",
        "url": str(url),
        "message": "Data returned by Flipkart Affiliate API.",
    }


def compare_marketplaces(query: str) -> List[Dict[str, Any]]:
    """Query both marketplaces without hiding missing credentials."""
    results = []
    try:
        results.append(search_amazon(query))
    except Exception as exc:
        results.append({
            "marketplace": "Amazon",
            "configured": True,
            "title": "Amazon lookup error",
            "price_display": "—",
            "availability": "Error",
            "offer": "—",
            "url": "",
            "message": str(exc),
        })

    try:
        results.append(search_flipkart(query))
    except Exception as exc:
        results.append({
            "marketplace": "Flipkart",
            "configured": True,
            "title": "Flipkart lookup error",
            "price_display": "—",
            "availability": "Error",
            "offer": "—",
            "url": "",
            "message": str(exc),
        })

    return results
