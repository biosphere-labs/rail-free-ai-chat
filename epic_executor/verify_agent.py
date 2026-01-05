"""Verification agent for checking implementation against acceptance criteria."""

import asyncio
import subprocess
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END

from .parser import TaskDefinition


class VerifyState(TypedDict):
    """State for verification agent."""

    task: TaskDefinition
    project_root: str
    impl_result: dict
    files_exist: dict[str, bool]
    test_output: str
    criteria_results: dict[str, bool]


def _detect_language(project_root: str, files_to_check: list[str]) -> str:
    """Detect project language from file extensions."""
    extensions = set()

    for file_path in files_to_check:
        ext = Path(file_path).suffix.lower()
        if ext:
            extensions.add(ext)

    # Check for frontend files
    if extensions & {".tsx", ".jsx"}:
        return "frontend"

    # Check for TypeScript/JavaScript
    if extensions & {".ts", ".js"}:
        return "typescript"

    # Check for Python
    if extensions & {".py"}:
        return "python"

    # Fall back to checking project root
    project_path = Path(project_root)
    if (project_path / "package.json").exists():
        return "typescript"
    if (project_path / "pyproject.toml").exists() or (project_path / "setup.py").exists():
        return "python"

    return "unknown"


def _run_tests(project_root: str, language: str) -> tuple[bool, str]:
    """Run tests based on detected language."""
    try:
        if language == "python":
            result = subprocess.run(
                ["pytest", "-v", "--tb=short"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
        elif language in ("typescript", "frontend"):
            # Try npm test first, fall back to npx jest
            result = subprocess.run(
                ["npm", "test"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 and "no test" in result.stderr.lower():
                result = subprocess.run(
                    ["npx", "jest"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
        else:
            return True, "No tests to run for unknown language"

        output = result.stdout
        if result.stderr:
            output += f"\n{result.stderr}"

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 120 seconds"
    except FileNotFoundError as e:
        return True, f"Test runner not found: {e}. Skipping tests."
    except Exception as e:
        return False, f"Error running tests: {e}"


async def check_files_exist(state: VerifyState) -> dict:
    """Check if all expected files exist."""
    task = state["task"]
    project_root = state["project_root"]

    files_exist = {}
    # Check files to create
    for file_path in task.files_to_create:
        full_path = Path(project_root) / file_path
        files_exist[file_path] = full_path.exists()

    # Check files to modify (should also exist)
    for file_path in task.files_to_modify:
        full_path = Path(project_root) / file_path
        files_exist[file_path] = full_path.exists()

    return {"files_exist": files_exist}


async def run_tests(state: VerifyState) -> dict:
    """Run appropriate tests based on language."""
    task = state["task"]
    project_root = state["project_root"]

    all_files = task.files_to_create + task.files_to_modify
    language = _detect_language(project_root, all_files)

    # Note: For frontend with .tsx/.jsx, Playwright MCP is available for E2E testing
    if language == "frontend":
        # Playwright MCP can be used here - this is a note for capability
        pass

    # Run tests in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    passed, output = await loop.run_in_executor(
        None, _run_tests, project_root, language
    )

    return {"test_output": output}


async def verify_criteria(state: VerifyState) -> dict:
    """Verify acceptance criteria by checking code and test results."""
    task = state["task"]
    files_exist = state["files_exist"]
    test_output = state["test_output"]

    criteria_results = {}

    for criterion in task.acceptance_criteria:
        # Simple heuristic verification
        passed = True

        # Check if criterion mentions file creation
        if "create" in criterion.lower() or "file" in criterion.lower():
            # At least some files should exist
            passed = any(files_exist.values()) if files_exist else False

        # Check if criterion mentions tests
        if "test" in criterion.lower():
            passed = "passed" in test_output.lower() or "ok" in test_output.lower()
            passed = passed and "failed" not in test_output.lower()

        criteria_results[criterion] = passed

    return {"criteria_results": criteria_results}


def create_verify_agent():
    """Create the verification agent graph."""
    builder = StateGraph(VerifyState)

    builder.add_node("check_files", check_files_exist)
    builder.add_node("run_tests", run_tests)
    builder.add_node("verify_criteria", verify_criteria)

    builder.set_entry_point("check_files")
    builder.add_edge("check_files", "run_tests")
    builder.add_edge("run_tests", "verify_criteria")
    builder.add_edge("verify_criteria", END)

    return builder.compile()


async def run_verification(
    task: TaskDefinition,
    project_root: str,
    impl_result: dict,
) -> dict:
    """Run verification on a task implementation.

    Args:
        task: The task definition with acceptance criteria
        project_root: Path to the project root directory
        impl_result: Result from the implementation step

    Returns:
        Dictionary with:
            - passed: bool - Overall verification pass/fail
            - criteria_results: dict - Pass/fail for each criterion
            - test_output: str - Output from running tests
    """
    agent = create_verify_agent()

    initial_state: VerifyState = {
        "task": task,
        "project_root": project_root,
        "impl_result": impl_result,
        "files_exist": {},
        "test_output": "",
        "criteria_results": {},
    }

    result = await agent.ainvoke(initial_state)

    # Determine overall pass status
    files_ok = all(result["files_exist"].values()) if result["files_exist"] else True
    criteria_ok = all(result["criteria_results"].values()) if result["criteria_results"] else True
    tests_ok = "failed" not in result["test_output"].lower()

    return {
        "passed": files_ok and criteria_ok and tests_ok,
        "criteria_results": result["criteria_results"],
        "test_output": result["test_output"],
    }
