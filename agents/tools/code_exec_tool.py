"""
Assembly code tracing / simulation tool.

Uses the LLM to simulate step-by-step execution of MIPS (or similar)
assembly code, showing register and memory state changes at each step.
Does NOT execute any code — all tracing is performed by the language model.
"""

from langchain_core.tools import tool


@tool
def trace_assembly_code(assembly_code: str) -> str:
    """Trace through MIPS assembly code step by step, showing register and memory changes.

    Use this tool when the user provides assembly code (MIPS, RISC-V, or
    similar) and wants to understand what it does, trace its execution,
    or see the register/memory state after each instruction.

    This tool uses the LLM to *simulate* execution — it does NOT run real
    machine code. It is best suited for short code snippets (< 30 instructions).

    Args:
        assembly_code: A string containing assembly instructions, one per line.
            Example:
                addi $t0, $zero, 5
                addi $t1, $zero, 3
                add $t2, $t0, $t1
                sw $t2, 0($sp)

    Returns:
        A step-by-step trace showing the state of registers and memory
        after each instruction, or an error message.
    """
    try:
        # Lazy import to avoid circular dependency
        from agents.llm import generate_with_chat_template

        # Validate input
        code = assembly_code.strip()
        if not code:
            return "Error: No assembly code provided. Please provide MIPS or RISC-V assembly instructions."

        lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(lines) > 50:
            return (
                f"Error: Code too long ({len(lines)} instructions). "
                "Please provide at most 50 instructions for accurate tracing."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a MIPS/RISC-V assembly simulator and Computer Architecture "
                    "teaching assistant. Your task is to trace through the given assembly "
                    "code step by step.\n\n"
                    "For EACH instruction, show:\n"
                    "1. The instruction being executed\n"
                    "2. What operation it performs (brief explanation)\n"
                    "3. The registers/memory that change and their new values\n\n"
                    "Formatting rules:\n"
                    "- Use a numbered list for each step\n"
                    "- Show register state as: $reg = value\n"
                    "- Show memory state as: MEM[address] = value\n"
                    "- After all steps, show a FINAL STATE summary table with all "
                    "  non-zero registers and any memory locations that were written\n"
                    "- If there is a loop, trace up to 3 full iterations, then summarise "
                    "  the pattern and give the final result\n\n"
                    "Assumptions (unless the code indicates otherwise):\n"
                    "- All registers start at 0 (except $sp = 0x7FFFFFFC)\n"
                    "- $zero is always 0 and cannot be written\n"
                    "- Memory is byte-addressable, word-aligned (4 bytes)\n"
                    "- Use decimal values by default, hex for addresses\n\n"
                    "If the assembly syntax is ambiguous or incorrect, note the issue "
                    "but attempt a best-effort trace anyway."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Trace the following assembly code step by step:\n\n"
                    f"```asm\n{code}\n```"
                ),
            },
        ]

        trace = generate_with_chat_template(messages, max_new_tokens=800)

        return (
            f"Assembly Code Trace\n"
            f"{'=' * 50}\n\n"
            f"Code ({len(lines)} instructions):\n"
            f"```\n{code}\n```\n\n"
            f"Step-by-Step Trace:\n"
            f"{'-' * 50}\n"
            f"{trace}"
        )

    except Exception as e:
        return f"Error tracing assembly code: {type(e).__name__}: {e}"
