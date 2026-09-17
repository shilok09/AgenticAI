"""
Reflex Agent — validates a stock analysis .txt file.
Returns PASS or FAIL with feedback.
"""

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY") and os.getenv("GROQ"):
    os.environ["GROQ_API_KEY"] = os.environ["GROQ"]

from agno.agent import Agent
from agno.models.groq import Groq


REFLEX_SYSTEM = """\
You are a validation agent. You check if a stock analysis is good.

Read the analysis and check:
1. Contains a stock price (dollar amount)
2. Contains a percentage change
3. Is 1-4 sentences (not a long report)
4. No made-up or placeholder data like [date] or [$price]

Output EXACTLY one line:
PASS: <brief reason>

Or if bad:
FAIL: <what's wrong and how to fix it>
"""


def validate_analysis(file_path: str) -> dict:
    """Read the .txt file and use LLM to validate the analysis."""
    if not os.path.exists(file_path):
        return {"pass": False, "feedback": f"File not found: {file_path}"}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return {"pass": False, "feedback": "File is empty."}

    # Quick sanity checks before calling LLM
    if "$" not in content and "price" not in content.lower():
        return {"pass": False, "feedback": "No price found in analysis."}

    if len(content) > 500:
        return {"pass": False, "feedback": f"Analysis too long ({len(content)} chars). Keep under 500."}

    # LLM validation
    agent = Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        instructions=REFLEX_SYSTEM,
        markdown=False,
    )

    response = agent.run(input=f"ANALYSIS TO VALIDATE:\n{content}")
    verdict = response.content.strip()

    passed = verdict.upper().startswith("PASS")
    return {"pass": passed, "feedback": verdict}


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reflex_agent.py <analysis_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = validate_analysis(file_path)

    print(f"\n{'='*60}")
    print(f"VALIDATING: {file_path}")
    print(f"{'='*60}")
    print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"FEEDBACK: {result['feedback']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
