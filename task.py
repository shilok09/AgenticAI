"""
Groq Chatbot — Structured Output
Model: openai/gpt-oss-120b via Groq

Usage:
    pip install groq python-dotenv
    python task.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
import os

load_dotenv()

from groq import Groq

client = Groq(api_key=os.environ["GROQ"])
MODEL = "openai/gpt-oss-120b"


def structured_output():
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": """Return a JSON object with exactly these keys for the person below:
- name
- age
- occupation

Person: John Smith, a 35-year-old software engineer.

Return ONLY the JSON, no explanation.""",
            }
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    structured_output()
