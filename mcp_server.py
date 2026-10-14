"""
MCP server exposing the agent's tools over the Model Context Protocol.

Run directly for stdio (what a local LangGraph client uses):
    python mcp_server.py

Run as an HTTP service (so other machines / clients can reach it):
    MCP_TRANSPORT=streamable-http python mcp_server.py
    -> http://127.0.0.1:8000/mcp

Inspect it by hand:
    npx @modelcontextprotocol/inspector python mcp_server.py
"""

import os

import anyio
import httpx
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")

HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

mcp = FastMCP(
    "mashfiq-tools",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)


# --------------------------------------------------------------------------
# The docstring of each function becomes the tool description the LLM reads,
# and the type hints become the JSON schema. Both matter for tool selection.
# --------------------------------------------------------------------------


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.

    Supported operations: "add", "sub", "mul", "div".
    """
    ops = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
    }

    if operation not in ops:
        return {"error": f"Unsupported operation '{operation}'. Use add, sub, mul or div."}

    if operation == "div" and second_num == 0:
        return {"error": "Division by zero is not allowed"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": ops[operation](first_num, second_num),
    }


@mcp.tool()
async def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web for current information and return the top results.

    Use this for anything time-sensitive or outside the model's knowledge.
    """
    def _search() -> list:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        # DDGS is blocking, so push it off the event loop.
        results = await anyio.to_thread.run_sync(_search)
        return {"query": query, "results": results}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@mcp.tool()
async def get_stock_price(symbol: str) -> dict:
    """Fetch the latest quote for a stock symbol, e.g. "AAPL" or "TSLA".

    Returns the current price, change, volume and previous close.
    """
    params = {
        "function": "GLOBAL_QUOTE",   # MARKET_STATUS ignores the symbol entirely
        "symbol": symbol.upper(),
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get("https://www.alphavantage.co/query", params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    # Alpha Vantage answers 200 OK with a Note/Information body when rate-limited.
    if "Note" in data or "Information" in data:
        return {"error": data.get("Note") or data.get("Information")}

    quote = data.get("Global Quote") or {}
    if not quote:
        return {"error": f"No quote returned for '{symbol}'. Check the symbol.", "raw": data}

    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "volume": quote.get("06. volume"),
        "previous_close": quote.get("08. previous close"),
        "latest_trading_day": quote.get("07. latest trading day"),
    }


@mcp.tool()
async def weather_update(city: str) -> dict:
    """Fetch the current weather for a city, e.g. "Dhaka" or "Tokyo"."""
    params = {"key": WEATHER_API_KEY, "q": city, "aqi": "no"}

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            r = await client.get("https://api.weatherapi.com/v1/current.json", params=params)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as exc:
        return {"error": f"Weather API returned {exc.response.status_code}", "body": exc.response.text[:300]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    location = data.get("location", {})
    current = data.get("current", {})

    return {
        "city": location.get("name"),
        "country": location.get("country"),
        "localtime": location.get("localtime"),
        "temp_c": current.get("temp_c"),
        "feelslike_c": current.get("feelslike_c"),
        "condition": (current.get("condition") or {}).get("text"),
        "humidity": current.get("humidity"),
        "wind_kph": current.get("wind_kph"),
    }


if __name__ == "__main__":
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))