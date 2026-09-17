"""
Simple ReAct Agent — Agno + yfinance
Manually implemented Think -> Act -> Observe loop with Reflex validation.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Agno expects GROQ_API_KEY; .env has GROQ
if not os.getenv("GROQ_API_KEY") and os.getenv("GROQ"):
    os.environ["GROQ_API_KEY"] = os.environ["GROQ"]

import yfinance as yf
from agno.agent import Agent
from agno.models.groq import Groq
from reflex_agent import validate_analysis


# ── Tools ───────────────────────────────────────────────────────────────────

def get_market_data(symbol: str) -> str:
    """Fetch basic market data for a stock ticker via yfinance."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="5d")

    if hist.empty:
        return json.dumps({"error": f"No data found for {symbol}"})

    latest = hist.iloc[-1]
    data_date = hist.index[-1].strftime("%Y-%m-%d %H:%M")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or latest["Close"]
    change = price - prev_close if prev_close else None
    change_pct = (change / prev_close * 100) if change is not None else None

    return json.dumps({
        "symbol": symbol.upper(),
        "data_timestamp": data_date,
        "price": round(price, 2),
        "previous_close": round(prev_close, 2) if prev_close else None,
        "change": round(change, 2) if change is not None else None,
        "change_pct": f"{change_pct:.2f}%" if change_pct is not None else None,
        "volume": int(latest["Volume"]),
        "market_cap": info.get("marketCap"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "name": info.get("shortName", symbol),
    }, indent=2)


def save_analysis(symbol: str, analysis: str) -> str:
    """Save analysis text to a .txt file. Returns the file path."""
    filename = f"analysis_{symbol.upper()}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(analysis)
    return os.path.abspath(filename)


# ── System prompt ───────────────────────────────────────────────────────────

def build_system_prompt(retries: int = 0, feedback: str = "") -> str:
    prompt = """\
You are a stock ticker analyzer. You have ONE job: call the tool, then summarize.

TOOL: get_market_data(symbol) -> returns real market data as JSON.

YOUR ONLY OUTPUT for step 1:
THINK: need data for SYMBOL
ACTION: get_market_data
INPUT: SYMBOL
DONE: false

After you see the OBSERVATION with real data, output step 2:
DONE: true
FINAL ANSWER: 2-3 sentence summary

NEVER make up data. NEVER include OBSERVATION in your response. Only output the format above.
"""
    if retries > 0 and feedback:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION. FIX THIS:\n{feedback}\n"
    return prompt


# ── Parse model output ─────────────────────────────────────────────────────

def parse_output(text: str) -> dict:
    r = {}

    m = re.search(r"THINK[:\s]+(.*?)(?=\n(?:ACTION|DONE|FINAL)\s*:)", text, re.DOTALL | re.IGNORECASE)
    if m:
        r["think"] = m.group(1).strip()

    m = re.search(r"ACTION[:\s]+(\w+)\s*(?:\(\s*['\"]?(\w+)['\"]?\s*\))?", text, re.IGNORECASE)
    if m:
        r["action"] = m.group(1).strip()
        if m.group(2):
            r["input"] = m.group(2).strip()

    if "action" not in r:
        m = re.search(r"(get_market_data)\s*\(\s*['\"]?(\w+)['\"]?\s*\)", text, re.IGNORECASE)
        if m:
            r["action"] = m.group(1).strip()
            r["input"] = m.group(2).strip()

    m = re.search(r"INPUT[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        r["input"] = m.group(1).strip()

    m = re.search(r"DONE[:\s]+(true|false)", text, re.IGNORECASE)
    if m:
        r["done"] = m.group(1).lower() == "true"

    m = re.search(r"(?:FINAL\s*ANSWER|Final\s*answer)[:\s]+(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        r["final"] = m.group(1).strip()

    return r


# ── ReAct loop ──────────────────────────────────────────────────────────────

def run_react(user_query: str, max_steps: int = 5, max_retries: int = 3) -> str:
    # Extract symbol from query (simple heuristic)
    words = user_query.upper().replace("ANALYZE", "").replace("FOR ME", "").split()
    symbol = words[0] if words else "AAPL"

    retry_count = 0
    feedback = ""

    for attempt in range(1, max_retries + 1):
        print(f"\n{'#'*60}")
        print(f"# ATTEMPT {attempt}/{max_retries}")
        print(f"{'#'*60}")

        if retry_count > 0:
            print(f"\n  RETRY FEEDBACK: {feedback}")

        agent = Agent(
            model=Groq(id="openai/gpt-oss-20b"),
            instructions=build_system_prompt(retry_count, feedback),
            markdown=False,
        )

        context = f"USER REQUEST: {user_query}\n\n"
        final_answer = None

        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"{'='*60}")

        for step in range(1, max_steps + 1):
            print(f"\n--- Step {step} ---")

            response = agent.run(input=context)
            model_text = response.content
            print(f"\nMODEL:\n{model_text}")

            parsed = parse_output(model_text)

            if parsed.get("think"):
                print(f"\n  THINK: {parsed['think']}")

            if parsed.get("action") == "get_market_data" and parsed.get("input"):
                sym = parsed["input"]
                print(f"\n  ACTION: get_market_data({sym})")

                observation = get_market_data(sym)
                print(f"\n  OBSERVATION:\n{observation}")

                context += f"STEP {step} RESPONSE:\n{model_text}\n\n"
                context += f"OBSERVATION from get_market_data({sym}):\n{observation}\n\n"
                context += "You now have the data. Give FINAL ANSWER in 2-3 sentences. Use format: DONE: true then FINAL ANSWER: <your text>\n\n"

            elif parsed.get("done") and parsed.get("final"):
                final_answer = parsed["final"]
                print(f"\n  FINAL ANSWER:\n{final_answer}")
                break

            elif "OBSERVATION from get_market_data" in context and not parsed.get("action"):
                answer = model_text.strip()
                if len(answer) > 20:
                    final_answer = answer
                    print(f"\n  FINAL ANSWER:\n{final_answer}")
                    break

            else:
                context += f"STEP {step} RESPONSE:\n{model_text}\n\n"
                context += "Invalid format. Use THINK/ACTION/INPUT/DONE or THINK/DONE/FINAL ANSWER.\n\n"

        if final_answer is None:
            print("\n  No final answer produced this attempt.")
            retry_count += 1
            feedback = "You did not produce a FINAL ANSWER. Always end with DONE: true then FINAL ANSWER."
            continue

        # ── Save to .txt ────────────────────────────────────────────────
        filepath = save_analysis(symbol, final_answer)
        print(f"\n{'='*60}")
        print(f"SAVED TO: {filepath}")
        print(f"{'='*60}")

        # ── Reflex validation ───────────────────────────────────────────
        print(f"\n--- Reflex Agent: validating ---")
        result = validate_analysis(filepath)
        print(f"  RESULT: {'PASS' if result['pass'] else 'FAIL'}")
        print(f"  FEEDBACK: {result['feedback']}")

        if result["pass"]:
            print(f"\n{'='*60}")
            print(f"VALIDATION PASSED")
            print(f"{'='*60}")
            print(f"\n{final_answer}")
            return final_answer

        # Failed — retry with feedback
        retry_count += 1
        feedback = result["feedback"]
        print(f"\n  VALIDATION FAILED. Will retry ({retry_count}/{max_retries})...")

    print(f"\n{'='*60}")
    print(f"ALL {max_retries} ATTEMPTS FAILED VALIDATION")
    print(f"{'='*60}")
    return "Analysis failed validation after maximum retries."


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Analyze AAPL for me"
    run_react(query)
