"""
Agno + NVIDIA Chatbot — Prompt Techniques & Temperature Testing
Model: minimaxai/minimax-m3 via NVIDIA NIM

Usage:
    pip install -U agno openai
    python agno_nvidia_chatbot.py
"""

import os
os.environ["NVIDIA_API_KEY"] = ""

from agno.agent import Agent
from agno.models.nvidia import Nvidia


def make_agent(temperature: float = 0.7, system_prompt: str = "You are a helpful assistant.") -> Agent:
    return Agent(
        model=Nvidia(
            id="minimaxai/minimax-m3",
            temperature=temperature,
        ),
        instructions=[system_prompt],
        markdown=True,
    )


def temperature_comparison():
    """Same prompt, different temperatures."""
    prompt = "Explain what a neural network is in 2-3 sentences."

    for temp in [0.0, 0.5, 1.0, 1.5]:
        print(f"\n{'='*60}")
        print(f"  TEMPERATURE = {temp}")
        print(f"{'='*60}")
        agent = make_agent(temperature=temp)
        agent.print_response(prompt)


def prompt_techniques():
    """Test different prompting strategies."""
    # 3a. Zero-Shot
    print("\n### Zero-Shot Prompting ###")
    agent = make_agent(temperature=0.3)
    agent.print_response(
        """Classify the following review as POSITIVE, NEGATIVE, or NEUTRAL:
"The product arrived on time but the quality was mediocre."

Respond with just the label."""
    )

    # 3b. Few-Shot
    print("\n### Few-Shot Prompting ###")
    agent.print_response(
        """Classify reviews as POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
- "Absolutely love this! Best purchase ever." -> POSITIVE
- "Terrible experience, never buying again." -> NEGATIVE
- "It works fine, nothing special." -> NEUTRAL

Now classify:
- "The product arrived on time but the quality was mediocre." ->"""
    )

    # 3c. Chain-of-Thought
    print("\n### Chain-of-Thought (CoT) Prompting ###")
    cot_agent = make_agent(temperature=0.2)
    cot_agent.print_response(
        """Solve this step by step:

A store sells notebooks for $4 each. If you buy 3 or more, you get 20% off.
How much does it cost to buy 5 notebooks?

Think through each step before giving the final answer."""
    )

    # 3d. Role-Playing
    print("\n### Role-Playing Prompt ###")
    role_agent = make_agent(
        temperature=0.5,
        system_prompt="You are a senior Python developer who gives concise, practical advice.",
    )
    role_agent.print_response("How should I handle exceptions in a production API?")

    # 3e. Structured Output
    print("\n### Structured Output Prompt ###")
    structured_agent = make_agent(temperature=0.1)
    structured_agent.print_response(
        """Return a JSON object with exactly these keys for the person below:
- name
- age
- occupation

Person: John Smith, a 35-year-old software engineer.

Return ONLY the JSON, no explanation."""
    )

    # 3f. Self-Consistency
    print("\n### Self-Consistency (Same Prompt, Multiple Runs) ###")
    agent_consistency = make_agent(temperature=0.9)
    print("Running 3 times at temp=0.9:")
    for i in range(3):
        print(f"\n--- Run {i+1} ---")
        agent_consistency.print_response("Answer with just the number:\nWhat is 15 + 27?")


def creative_sweep():
    """Creative vs Precise — Temperature Sweep."""
    creative_prompt = "Write a one-sentence story about a robot."

    for temp in [0.0, 0.7, 1.2]:
        label = "precise" if temp < 0.5 else "balanced" if temp < 1.0 else "creative"
        print(f"\n{'='*60}")
        print(f"  TEMP = {temp}  ({label})")
        print(f"{'='*60}")
        a = make_agent(temperature=temp)
        a.print_response(creative_prompt)


def interactive_chat():
    """Interactive chatbot. Type 'quit' to exit."""
    chat_agent = make_agent(temperature=0.7)

    print("Interactive chatbot (type 'quit' to exit):")
    print("-" * 40)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        chat_agent.print_response(user_input)


if __name__ == "__main__":
    print("=" * 60)
    print("  TEMPERATURE COMPARISON")
    print("=" * 60)
    temperature_comparison()

    print("\n" + "=" * 60)
    print("  PROMPT TECHNIQUES")
    print("=" * 60)
    prompt_techniques()

    print("\n" + "=" * 60)
    print("  CREATIVE vs PRECISE")
    print("=" * 60)
    creative_sweep()

    print("\n" + "=" * 60)
    print("  INTERACTIVE CHAT")
    print("=" * 60)
    interactive_chat()
