"""
Planner Agent — LangChain + ChatGroq
5 tools: search_flights, search_hotels, get_weather, save_plan_pdf, send_plan_email
"""

import os
import json
import re
import urllib.request
import urllib.parse
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from pdf_tool import markdown_to_pdf
from email_tool import send_plan_email

load_dotenv()

if not os.getenv("GROQ_API_KEY") and os.getenv("GROQ"):
    os.environ["GROQ_API_KEY"] = os.environ["GROQ"]


# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search real flights or buses between two cities on a given date using DuckDuckGo.
    Args:
        origin: Departure city (e.g. 'Lahore', 'Islamabad')
        destination: Arrival city (e.g. 'Murree', 'Nathia Gali')
        date: Travel date in YYYY-MM-DD format
    Returns:
        JSON string with flight/bus options found via web search.
    """
    query = f"how to travel from {origin} to {destination} Pakistan bus flight ticket price 2026"
    raw = ddg_search.invoke(query)

    results = []
    seen = set()
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue

        title_m = re.search(r"title:\s*(.+?)(?:,\s*link:|$)", line)
        snippet_m = re.search(r"snippet:\s*(.+?)(?:,\s*title:|$)", line)
        link_m = re.search(r"link:\s*(\S+)", line)

        title = title_m.group(1).strip() if title_m else ""
        snippet = snippet_m.group(1).strip() if snippet_m else ""
        url = link_m.group(1).strip() if link_m else ""

        if not title or title.lower() in seen:
            continue

        combined = (title + " " + snippet).lower()
        if not any(kw in combined for kw in ["flight", "bus", "airline", "airblue", "pia", "faisal", "daewoo", "road", "travel"]):
            continue

        seen.add(title.lower())

        # Extract price
        price = None
        for pat in [r"PKR\s*([\d,]+)", r"from\s*\$(\d+)", r"\$(\d+)", r"Rs\.?\s*([\d,]+)"]:
            pm = re.search(pat, snippet, re.IGNORECASE)
            if pm:
                val = int(pm.group(1).replace(",", ""))
                price = val * 280 if "PKR" not in pat and "Rs" not in pat else val
                break

        results.append({
            "name": title[:80],
            "snippet": snippet[:200],
            "price_hint": f"PKR {price}" if price else "check source",
            "source_url": url,
        })

    if not results:
        results = [{"note": f"No transport results found for {origin} to {destination}. Consider bus services like Daewoo or Faisal Movers."}]

    return json.dumps(results[:6], indent=2)


ddg_search = DuckDuckGoSearchResults(max_results=10)


@tool
def search_hotels(city: str, checkin: str, checkout: str, budget_max: int) -> str:
    """Search real hotels in a city using DuckDuckGo web search.
    Args:
        city: Destination city name
        checkin: Check-in date in YYYY-MM-DD format
        checkout: Check-out date in YYYY-MM-DD format
        budget_max: Maximum price per night in PKR
    Returns:
        JSON string with hotel names, sources, and links found via web search.
    """
    query = f"best hotels in {city} Pakistan prices per night PKR 2026"
    raw = ddg_search.invoke(query)

    hotels = []
    seen = set()

    for line in raw.strip().split("\n"):
        if not line.strip():
            continue

        title_m = re.search(r"title:\s*(.+?)(?:,\s*link:|$)", line)
        snippet_m = re.search(r"snippet:\s*(.+?)(?:,\s*title:|$)", line)
        link_m = re.search(r"link:\s*(\S+)", line)

        title = title_m.group(1).strip() if title_m else ""
        snippet = snippet_m.group(1).strip() if snippet_m else ""
        url = link_m.group(1).strip() if link_m else ""

        if not title or title.lower() in seen:
            continue

        combined = (title + " " + snippet).lower()
        if not any(kw in combined for kw in ["hotel", "resort", "lodge", "inn", "guest", "hostel", "accommodation"]):
            continue

        seen.add(title.lower())

        # Try to extract any price mentioned
        price = None
        for pat in [r"PKR\s*([\d,]+)", r"from\s*\$(\d+)", r"\$(\d+)\s*(?:per|/)\s*night"]:
            pm = re.search(pat, snippet, re.IGNORECASE)
            if pm:
                val = int(pm.group(1).replace(",", ""))
                price = val * 280 if "PKR" not in pat else val
                break

        hotels.append({
            "name": title[:80],
            "snippet": snippet[:200],
            "price_hint": f"PKR {price}" if price else "check source",
            "source_url": url,
        })

    if not hotels:
        hotels = [{"note": f"No hotel results found for {city}."}]

    return json.dumps(hotels[:8], indent=2)


@tool
def get_weather(city: str, date: str) -> str:
    """Get real weather forecast for a city on a given date using Open-Meteo API.
    Args:
        city: City name (e.g. 'Murree', 'Lahore', 'Paris')
        date: Date in YYYY-MM-DD format (must be within 16 days of today)
    Returns:
        JSON string with temperature, conditions, precipitation, wind, sunrise/sunset.
    """
    WMO_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Freezing drizzle (light)", 57: "Freezing drizzle (dense)",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Freezing rain (light)", 67: "Freezing rain (heavy)",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
    }

    # Step 1: Geocode city name -> lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1}
    )
    try:
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo_data = json.loads(resp.read().decode())
        if not geo_data.get("results"):
            return json.dumps({"error": f"Could not geocode city: {city}"})
        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = loc.get("name", city)
        country = loc.get("country", "")
    except Exception as e:
        return json.dumps({"error": f"Geocoding failed: {e}"})

    # Step 2: Fetch daily forecast (try requested date, fall back to default if out of range)
    daily_fields = "temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,sunrise,sunset"
    base = {"latitude": lat, "longitude": lon, "daily": daily_fields, "timezone": "auto"}

    # Try with explicit date range first
    params = {**base, "start_date": date, "end_date": date}
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(forecast_url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError:
        # Date out of range — fall back to default 7-day forecast
        forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(base)
        try:
            with urllib.request.urlopen(forecast_url, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            return json.dumps({"error": f"Forecast API failed: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Forecast API failed: {e}"})

    daily = data.get("daily", {})
    if not daily or not daily.get("time"):
        return json.dumps({"error": "No forecast data returned"})

    weather_code = daily["weathercode"][0]
    weather = {
        "city": resolved_name,
        "country": country,
        "date_requested": date,
        "date_returned": daily["time"][0],
        "latitude": lat,
        "longitude": lon,
        "temperature": {
            "high_c": daily["temperature_2m_max"][0],
            "low_c": daily["temperature_2m_min"][0],
        },
        "conditions": WMO_CODES.get(weather_code, f"Code {weather_code}"),
        "precipitation_mm": daily["precipitation_sum"][0],
        "precipitation_chance_%": daily["precipitation_probability_max"][0],
        "wind_speed_kmh": daily["wind_speed_10m_max"][0],
        "sunrise": daily["sunrise"][0],
        "sunset": daily["sunset"][0],
    }
    return json.dumps(weather, indent=2)


@tool
def save_plan_pdf(markdown_text: str, filename: str) -> str:
    """Save the trip plan as a styled PDF file.
    Args:
        markdown_text: The full plan in markdown format
        filename: Output PDF filename (e.g. 'trip_plan.pdf')
    Returns:
        Absolute path to the saved PDF file.
    """
    path = markdown_to_pdf(markdown_text, filename)
    return f"PDF saved to: {path}"


@tool
def send_plan_email_tool(pdf_path: str, receiver_email: str) -> str:
    """Send the trip plan PDF via email to a recipient.
    Args:
        pdf_path: Absolute path to the PDF file
        receiver_email: Recipient's email address
    Returns:
        Success or error message.
    """
    result = send_plan_email(receiver_email, pdf_path)
    return result


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM = """\
You are a TRIP PLANNER agent with 5 tools. ALWAYS call tools before planning.

Tools:
- search_flights(origin, destination, date) -> flight/bus options with prices
- search_hotels(city, checkin, checkout, budget_max) -> hotel options with prices
- get_weather(city, date) -> weather forecast
- save_plan_pdf(markdown_text, filename) -> saves plan as PDF
- send_plan_email_tool(pdf_path, receiver_email) -> sends PDF via email

WORKFLOW:
1. Parse user goal for: origin, destination, dates, budget.
2. Call search_flights, search_hotels, get_weather with the data you have.
3. Write the plan in markdown format with REAL prices from tool results.
4. After writing the plan, call save_plan_pdf to save it.
5. THEN ask the user: "What is the receiver email address?"
6. When user provides the email, call send_plan_email_tool to send it.

IMPORTANT: When you reach step 4-5, STOP and ask for the email. Do NOT call
send_plan_email_tool until the user provides an email address.

PLAN FORMAT:
GOAL: <one line>
BUDGET BREAKDOWN:
- Transport: PKR X (bus/flight details)
- Hotel: PKR X/night x N nights = PKR Y (hotel name)
- Total: PKR Z / Budget: PKR W

ITINERARY:
DAY 1: <activities with times>
DAY 2: ...
DAY 3: ...
"""


# ── Agent loop ──────────────────────────────────────────────────────────────

def run_planner(goal: str):
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    tools = [search_flights, search_hotels, get_weather, save_plan_pdf, send_plan_email_tool]
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=goal),
    ]

    print(f"\n{'='*60}")
    print(f"GOAL: {goal}")
    print(f"{'='*60}")

    for step in range(1, 15):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If model made tool calls, execute them
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                print(f"\n  TOOL CALL: {tool_name}({tool_args})")

                tool_map = {t.name: t for t in tools}
                result = tool_map[tool_name].invoke(tool_args)
                print(f"  RESULT: {result[:300]}...")

                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # If no tool calls, the model is done — print the plan or ask for email
        elif response.content:
            print(f"\n{'='*60}")
            print(response.content)
            print(f"{'='*60}")

            # Check if this looks like the final plan (has BUDGET BREAKDOWN)
            if "BUDGET BREAKDOWN" in response.content or "ITINERARY" in response.content:
                filename = "plan_langchain.pdf"
                abs_path = markdown_to_pdf(response.content, filename)
                print(f"\n  PDF SAVED TO: {abs_path}")

                # Save markdown too
                with open("plan_langchain.txt", "w", encoding="utf-8") as f:
                    f.write(response.content)

                # Ask for email
                email = input("\n  ENTER RECEIVER EMAIL ADDRESS: ").strip()
                if email:
                    email_result = send_plan_email(abs_path, email)
                    print(f"\n  {email_result}")
                else:
                    print("  No email provided. PDF saved locally.")
                return response.content

    return "Max steps reached."


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not goal:
        goal = input("ENTER YOUR GOAL: ").strip()
    if not goal:
        print("No goal provided. Exiting.")
        sys.exit(1)
    run_planner(goal)
