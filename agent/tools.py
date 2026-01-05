import subprocess
import sys
import tempfile
from typing import Literal
from langchain_core.tools import tool


@tool
def execute_code(
    code: str,
    language: Literal["python", "shell"] = "python",
) -> str:
    """Execute code locally on the user's machine.

    Args:
        code: The code to execute.
        language: The language of the code - either "python" or "shell".

    Returns:
        The output of the code execution (stdout and stderr combined).
    """
    try:
        if language == "python":
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
        else:  # shell
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        return output if output.strip() else "[No output]"

    except subprocess.TimeoutExpired:
        return "[Error: Code execution timed out after 60 seconds]"
    except Exception as e:
        return f"[Error: {type(e).__name__}: {e}]"


# List of tools for the agent
def get_tools():
    """Get all available tools for the agent."""
    return [execute_code]
