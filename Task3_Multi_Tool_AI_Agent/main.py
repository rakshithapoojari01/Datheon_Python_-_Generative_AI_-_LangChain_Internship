
import os
import ast
import operator
from pathlib import Path

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Please add it to Task3_Multi_Tool_AI_Agent/.env"
    )


# --------------------------------------------------
# 2. WEB SEARCH TOOL
# --------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No web search results found."

        output = []

        for item in results:
            output.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('href', '')}\n"
                f"Summary: {item.get('body', '')}"
            )

        return "\n\n".join(output)

    except Exception as error:
        return f"Web search failed: {error}"


# --------------------------------------------------
# 3. PYTHON MATH CALCULATOR TOOL
# --------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic expressions safely."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.operand)
            )

        raise ValueError(
            "Only basic arithmetic expressions are allowed."
        )

    try:
        parsed_expression = ast.parse(expression, mode="eval")
        result = evaluate(parsed_expression.body)
        return str(result)

    except Exception as error:
        return f"Calculation error: {error}"


# --------------------------------------------------
# 4. LOCAL FILE READER TOOL
# --------------------------------------------------

@tool
def read_local_file(filename: str) -> str:
    """Read a text file located inside the Task 3 project folder."""

    try:
        file_path = (BASE_DIR / filename).resolve()

        # Prevent access outside the project folder.
        if not file_path.is_relative_to(BASE_DIR):
            return "Access denied: choose a file inside the Task 3 folder."

        if not file_path.is_file():
            return f"File not found: {filename}"

        return file_path.read_text(encoding="utf-8")[:10000]

    except Exception as error:
        return f"Could not read file: {error}"


# --------------------------------------------------
# 5. REGISTER TOOLS
# --------------------------------------------------

TOOLS = [
    web_search,
    calculator,
    read_local_file,
]

print("Tools loaded successfully:")

for available_tool in TOOLS:
    print(f"- {available_tool.name}")


# --------------------------------------------------
# 6. INITIALISE GEMINI
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=API_KEY,
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=(
        "You are a helpful multi-tool autonomous AI agent. "
        "Use web_search for current information, calculator for "
        "mathematical calculations, and read_local_file to read "
        "local project files. Invoke the appropriate tools when "
        "needed. Base your final answers on tool observations. "
        "Do not claim to have used a tool unless you actually did."
    ),
)


# --------------------------------------------------
# 7. INTERACTIVE AGENT LOOP
# --------------------------------------------------

print("\nMulti-Tool AI Agent is ready!")
print("Available tools: Web Search, Calculator, Local File Reader")
print("Type 'exit' to stop.\n")


while True:
    try:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if not query:
            continue

        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        messages = response["messages"]

        print("\n--- Agent Execution Trace ---")

        for message in messages:
            tool_calls = getattr(message, "tool_calls", [])

            if tool_calls:
                for call in tool_calls:
                    print(f"\nAction: {call['name']}")
                    print(f"Input: {call['args']}")

            if getattr(message, "type", "") == "tool":
                observation = str(message.content)
                print(f"Observation: {observation[:300]}")

        answer = messages[-1].content

        if isinstance(answer, list):
            answer = "\n".join(
                item["text"]
                for item in answer
                if isinstance(item, dict)
                and item.get("type") == "text"
            )

        print("\n--- Final Answer ---")
        print("Agent:", answer)
        print()

    except KeyboardInterrupt:
        print("\nAgent stopped.")
        break

    except Exception as error:
        print(f"\nAgent error: {error}\n")
