import json
import os
import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON

from config import (
    MODEL, MAX_LOOP_ITERATIONS, MAX_REFUND_AMOUNT,
    CYAN, GREEN, YELLOW, RED, DIM, BOLD, RESET,
)
from tools import ALL_TOOLS, TOOL_FUNCTIONS

load_dotenv()
console = Console()


def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(prompt_path, "r") as f:
        content = f.read()
    return content


# --- Prerequisite gate ---
# [Task 1.4] — programmatic enforcement: block process_refund until get_customer has run

def check_prerequisite(tool_name, tool_history):
    """SOLVED: Blocks process_refund unless get_customer ran first."""
    if tool_name == "process_refund" and "get_customer" not in tool_history:
        return {
            "error": True,
            "errorCategory": "validation",
            "isRetryable": True,
            "message": "CRITICAL: You must call 'get_customer' to verify identity before processing a refund."
        }
    return None


# --- PostToolUse hook ---
# [Task 1.5] — hook intercepts outgoing tool calls for compliance enforcement

def post_tool_use_hook(tool_name, tool_input, tool_result):
    """SOLVED: Intercepts high-value refunds and auto-escalates."""
    if tool_name == "process_refund" and float(tool_input.get("amount", 0)) > MAX_REFUND_AMOUNT:
        return {
            "error": True,
            "errorCategory": "policy_violation",
            "isRetryable": False,
            "message": f"Refund amount ${tool_input['amount']} exceeds the ${MAX_REFUND_AMOUNT} auto-approve limit.",
            "action": "escalate_to_human"
        }
    return tool_result


# --- Tool execution ---

def execute_tool(tool_name, tool_input, tool_history):
    """Execute a tool call with prerequisite check and post-use hook."""
    # check prerequisite gate before execution
    gate_result = check_prerequisite(tool_name, tool_history)
    if gate_result is not None:
        print(f"  {RED}⛔ Gate blocked {tool_name}: prerequisite not met{RESET}")
        return gate_result

    # Execute the actual lab tool function from tools.py
    tool_fn = TOOL_FUNCTIONS.get(tool_name)
    if not tool_fn:
        return {
            "error": True,
            "errorCategory": "validation",
            "isRetryable": False,
            "message": f"Unknown tool: {tool_name}",
        }

    raw_result = tool_fn(**tool_input)

    # apply PostToolUse hook after execution
    result = post_tool_use_hook(tool_name, tool_input, raw_result)

    # Track which tools have been called successfully
    if not (isinstance(result, dict) and result.get("error")):
        tool_history.add(tool_name)

    return result


# --- Agentic loop ---

def run_agent(user_message):
    """Run the agentic loop for a single user message."""
    client = anthropic.Anthropic()
    system_template = load_system_prompt()

    messages = [{"role": "user", "content": user_message}]
    tool_history = set()
    case_facts = "Verified customer interaction context."

    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"Customer: {user_message}")
    print(f"{'='*60}{RESET}")

    for iteration in range(MAX_LOOP_ITERATIONS):
        print(f"\n{DIM}--- Iteration {iteration + 1} ---{RESET}")

        # format system prompt template with current case_facts
        system_prompt = system_template.format(case_facts=case_facts)
        print(f"  {DIM}case_facts: {case_facts}{RESET}")

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system_prompt,
                tools=ALL_TOOLS,
                messages=messages,
            )
        except Exception as e:
            print(f"\n{RED}{BOLD}API Error: {e}{RESET}")
            return

        print(f"  {DIM}stop_reason: {response.stop_reason}{RESET}")

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\n{GREEN}{BOLD}Agent:{RESET} {GREEN}{block.text}{RESET}")
            break

        if response.stop_reason == "tool_use":
            tool_result_blocks = []

            for block in response.content:
                if block.type == "tool_use":
                    print(f"  {YELLOW}Tool call: {block.name}{RESET}")
                    console.print(JSON(json.dumps(block.input, indent=2)), style="dim")

                    result = execute_tool(block.name, block.input, tool_history)
                    print(f"  {DIM}Result:{RESET}")
                    console.print(JSON(json.dumps(result, indent=2)), style="dim")

                    tool_result_block = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                    tool_result_blocks.append(tool_result_block)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_result_blocks})


def main():
    test_queries = [
        "Hi, it's Asha. My blender stopped working, I want a refund.",
        "I bought a water heater from you, it's leaking, refund please.",
        "Refund my $1,200 laptop."
    ]

    print(f"{BOLD}Lab 01 — Customer Support Resolution Agent{RESET}\n")
    for i, query in enumerate(test_queries, 1):
        print(f"  {DIM}{i}. {query}{RESET}")

    while True:
        user_input = input(f"\n{CYAN}Customer > {RESET}").strip()

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.isdigit() and 1 <= int(user_input) <= len(test_queries):
            user_input = test_queries[int(user_input) - 1]
            print(f"  {DIM}→ {user_input}{RESET}")

        run_agent(user_input)


if __name__ == "__main__":
    main()