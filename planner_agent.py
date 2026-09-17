"""
Planner Agent — Agno
Goal-decomposition for travel planning with structured step output.
"""

import json
import os
import re
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY") and os.getenv("GROQ"):
    os.environ["GROQ_API_KEY"] = os.environ["GROQ"]

from agno.agent import Agent
from agno.models.groq import Groq


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM = """\
You are a TRIP PLANNER. You produce a plan for an executor. You never book anything.

STEP 1: Read the user goal. Extract what you can (destination, dates, budget, travelers).
STEP 2: If ANY required slot is missing, ask ALL missing slots in ONE message (up to 3 questions).
STEP 3: Once all slots are filled, output the plan immediately.

REQUIRED SLOTS (you MUST know all before planning):
- TRAVELERS: count (solo=1 adult, couple=2 adults, family=adults+children with ages)
- ORIGIN: departure city
- DESTINATION: where
- DATES: when (exact dates or "3 days from now")
- BUDGET: how much + currency
- ACCOMMODATION: hotel / hostel / apartment
- TRANSPORT: car / bus / train / flight

IF YOU ALREADY KNOW IT FROM THE USER GOAL, DO NOT ASK AGAIN.
Example: user says "solo trip" -> travelers = 1 adult. DO NOT ask "how many travelers?"

--- Question format (ask ALL missing in one message) ---
QUESTION 1: <slot>
OPTIONS:
- option A
- option B

QUESTION 2: <slot>
OPTIONS:
- option A
- option B

--- Plan format ---
GOAL: <one line summary>
ASSUMPTIONS:
- <any assumptions made>

STEP 1: <single action>
  ACTION: <what to do>
  INPUT: <what this step needs>
  OUTPUT: <what this step produces>
  VERIFY: <how to check it worked>
  ON FAILURE: <what to do if it fails>

STEP 2: ...

DONE WHEN: <completion condition>

RULES:
- Every step = ONE action. No "and then".
- No banned words: research, consider, explore, arrange, suitable, various.
- Every step has ACTION, INPUT, OUTPUT, VERIFY, ON FAILURE.
- Use exact values from user answers. Never use placeholders like <city>.
- Steps are in dependency order. Mark [PARALLEL-OK] if independent.
"""


# ── Parse agent output ──────────────────────────────────────────────────────

def is_question_line(line: str) -> bool:
    """Check if a line is a question (ends with ? or starts with question word)."""
    stripped = line.strip()
    if stripped.endswith("?"):
        return True
    if re.match(r"^(are you|what is|what are|how many|which|where|when|do you|can you|would you)", stripped, re.IGNORECASE):
        return True
    return False


def parse_agent_output(text: str) -> dict:
    r = {}

    # Detect question blocks: "QUESTION N:" or just "QUESTION:"
    question_blocks = re.findall(
        r"(?:QUESTION\s*\d*[:\s]+)(.+?)(?=QUESTION\s*\d*[:\s]+|PLAN:|GOAL:|STEP\s+\d+|\Z)",
        text, re.DOTALL | re.IGNORECASE
    )

    # Also detect if the output is ONLY questions (no STEP/PLAN/GOAL)
    has_step = bool(re.search(r"(?:^|\n)\s*STEP\s+\d+", text, re.IGNORECASE))
    has_goal = bool(re.search(r"GOAL:", text, re.IGNORECASE))

    if question_blocks and not has_step and not has_goal:
        r["type"] = "question"
        # Parse the first question block
        first_block = question_blocks[0].strip()
        q_match = re.match(r"(.+?)(?:\nOPTIONS:|\Z)", first_block, re.DOTALL | re.IGNORECASE)
        r["question"] = q_match.group(1).strip() if q_match else first_block

        # Parse all options from the full text
        opts = re.findall(r"^-\s+(.+)$", text, re.MULTILINE)
        r["options"] = [o.strip() for o in opts if "other" not in o.lower()]
        r["all_questions"] = [q.strip() for q in question_blocks]
        return r

    # ── Plan: parse structured steps ──
    steps = []

    step_blocks = re.split(r"(?=^\s*STEP\s+\d+)", text, flags=re.MULTILINE | re.IGNORECASE)

    for block in step_blocks:
        block = block.strip()
        step_match = re.match(r"STEP\s+(\d+)[:\s]+(.+)", block, re.IGNORECASE)
        if not step_match:
            continue

        step_num = int(step_match.group(1))
        step_title = step_match.group(2).strip()

        # Skip if the "step" is actually a question
        if is_question_line(step_title):
            continue

        step_data = {"num": step_num, "title": step_title}

        for field in ["ACTION", "INPUT", "OUTPUT", "VERIFY", "ON FAILURE"]:
            fm = re.search(
                rf"{field}[:\s]+(.+?)(?=\n\s*(?:ACTION|INPUT|OUTPUT|VERIFY|ON FAILURE|STEP|\Z))",
                block, re.DOTALL | re.IGNORECASE
            )
            if fm:
                step_data[field.lower().replace(" ", "_")] = fm.group(1).strip()

        steps.append(step_data)

    # Fallback: numbered list
    if not steps:
        for m in re.finditer(r"^\s*(\d+)[\.\)]\s+(.+)", text, re.MULTILINE):
            title = m.group(2).strip()
            if not is_question_line(title):
                steps.append({"num": int(m.group(1)), "title": title})

    if steps:
        r["type"] = "plan"
        r["steps"] = steps

        g = re.search(r"GOAL[:\s]+(.+?)(?=\nASSUMPTIONS:|\nSTEP|\Z)", text, re.DOTALL | re.IGNORECASE)
        r["goal"] = g.group(1).strip() if g else ""

        a = re.search(r"ASSUMPTIONS:\s*\n((?:-.+\n?)+)", text, re.IGNORECASE)
        r["assumptions"] = [l.strip().lstrip("- ") for l in a.group(1).splitlines() if l.strip()] if a else []

        d = re.search(r"DONE WHEN[:\s]+(.+)", text, re.DOTALL | re.IGNORECASE)
        r["done_when"] = d.group(1).strip() if d else ""

        return r

    r["type"] = "unknown"
    r["raw"] = text
    return r


# ── Display ─────────────────────────────────────────────────────────────────

def print_plan(parsed: dict):
    print(f"\n{'='*60}")
    print(f"GOAL: {parsed.get('goal', 'N/A')}")
    print(f"{'='*60}")

    if parsed.get("assumptions"):
        print("\nASSUMPTIONS:")
        for a in parsed["assumptions"]:
            print(f"  - {a}")

    for step in parsed["steps"]:
        num = step.get("num", "?")
        title = step.get("title", "")
        print(f"\n  STEP {num}: {title}")
        for field in ["action", "input", "output", "verify", "on_failure"]:
            if field in step:
                label = field.replace("_", " ").upper()
                print(f"    {label}: {step[field]}")

    if parsed.get("done_when"):
        print(f"\nDONE WHEN: {parsed['done_when']}")

    print(f"\n{'='*60}")


def save_plan(parsed: dict) -> str:
    filename = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"GOAL: {parsed.get('goal', 'N/A')}\n\n")

        if parsed.get("assumptions"):
            f.write("ASSUMPTIONS:\n")
            for a in parsed["assumptions"]:
                f.write(f"  - {a}\n")
            f.write("\n")

        for step in parsed["steps"]:
            num = step.get("num", "?")
            title = step.get("title", "")
            f.write(f"STEP {num}: {title}\n")
            for field in ["action", "input", "output", "verify", "on_failure"]:
                if field in step:
                    label = field.replace("_", " ").upper()
                    f.write(f"  {label}: {step[field]}\n")
            f.write("\n")

        if parsed.get("done_when"):
            f.write(f"DONE WHEN: {parsed['done_when']}\n")

    return os.path.abspath(filename)


# ── Interactive loop ────────────────────────────────────────────────────────

import questionary

CLEAR = "\033[2J\033[H"  # clear screen + move cursor to top


def clear_screen():
    print(CLEAR, end="", flush=True)


def ask_single(question: str, options: list[str]) -> str:
    """Interactive arrow-key selector for one question."""
    if not options:
        return questionary.text(question).ask() or ""

    try:
        choice = questionary.select(
            question,
            choices=options + ["Other (type my own answer)"],
        ).ask()
        if choice is None:
            choice = options[0]
        if "Other" in choice:
            return questionary.text("Type your answer:").ask() or options[0]
        return choice
    except Exception:
        # Fallback: numbered input
        print(f"\n  {question}")
        for i, opt in enumerate(options, 1):
            print(f"    [{i}] {opt}")
        while True:
            raw = input("  PICK NUMBER (or type answer): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            if raw:
                return raw


def ask_user_batch(text: str, questions: list[str]) -> dict[str, str]:
    """Ask multiple questions with arrow-key selection. Clears screen between each."""
    answers = {}
    clear_screen()
    print("=" * 50)
    print("  AGENT NEEDS A FEW DETAILS")
    print("=" * 50)

    for i, q in enumerate(questions, 1):
        clear_screen()
        print(f"  Question {i}/{len(questions)}")
        print(f"  {q}")
        print()

        # Extract options for this question from the full text
        opts = extract_options_for_q(text, q)
        answer = ask_single(f"  {q}", opts)
        answers[q] = answer
        print(f"  -> {answer}")

    return answers


def extract_options_for_q(full_text: str, question: str) -> list[str]:
    """Extract OPTIONS that immediately follow a specific QUESTION in the agent output."""
    # Find this question, then grab options until next QUESTION or end
    idx = full_text.find(question[:40])
    if idx == -1:
        return []
    # Search for OPTIONS after this question
    chunk = full_text[idx:idx + 500]
    m = re.search(r"OPTIONS:\s*\n((?:-\s+.+\n?)+)", chunk, re.IGNORECASE)
    if m:
        opts = re.findall(r"^-\s+(.+)$", m.group(1), re.MULTILINE)
        return [o.strip() for o in opts if "other" not in o.lower()]
    return []


def run_planner(goal: str, max_question_rounds: int = 1) -> str:
    """Main loop: max 1 question round + 1 plan round = 2 rounds total."""
    agent = Agent(
        model=Groq(id="openai/gpt-oss-20b"),
        instructions=SYSTEM,
        markdown=False,
    )

    context = f"USER GOAL: {goal}\n\n"

    clear_screen()
    print("=" * 50)
    print(f"  GOAL: {goal}")
    print("=" * 50)

    questions_asked = 0

    for round_num in range(1, 4):
        response = agent.run(input=context)
        agent_text = response.content

        parsed = parse_agent_output(agent_text)

        # ── Question ──
        if parsed.get("type") == "question" and questions_asked < max_question_rounds:
            questions = parsed.get("all_questions", [parsed["question"]])[:3]
            questions_asked += 1

            answers = ask_user_batch(agent_text, questions)

            for q, a in answers.items():
                context += f"AGENT QUESTION: {q}\nUSER ANSWER: {a}\n\n"

            continue

        # ── Plan ──
        elif parsed.get("type") == "plan" and parsed.get("steps"):
            clear_screen()
            print_plan(parsed)
            filepath = save_plan(parsed)
            print(f"\n  SAVED TO: {filepath}")
            return parsed.get("goal", goal)

        # ── Unknown — force plan ──
        else:
            context += (
                f"AGENT RESPONSE: {agent_text}\n\n"
                "You must output a PLAN now. Do not ask more questions. "
                "Output GOAL, ASSUMPTIONS, STEP 1... with ACTION/INPUT/OUTPUT/VERIFY/ON FAILURE.\n\n"
            )

    print("\n  ERROR: Agent could not produce a valid plan.")
    return "Failed to produce plan."


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not goal:
        goal = questionary.text("ENTER YOUR GOAL:").ask()
    if not goal:
        print("No goal provided. Exiting.")
        sys.exit(1)
    run_planner(goal)
