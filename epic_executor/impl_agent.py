"""Implementation Agent for executing task definitions using LangGraph."""

import os
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .parser import TaskDefinition
from agent.tools import execute_code


IMPL_SYSTEM_PROMPT = """You are an implementation agent that executes coding tasks.

You have access to file operations and code execution tools. Your job is to:
1. Read the task definition carefully
2. Implement the required deliverables
3. Ensure all acceptance criteria are met
4. Create or modify the specified files

Work methodically:
- First read any existing files you need to modify
- Plan your implementation
- Write/modify files as needed
- Test your changes if possible
- Report what you've done

Be precise and complete. Follow the acceptance criteria exactly."""


class ImplAgentState(TypedDict):
    """State for the implementation agent."""
    messages: Annotated[list, add_messages]
    files_modified: list[str]


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file to read (relative to project root or absolute).

    Returns:
        The file contents as a string, or an error message if the file cannot be read.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[Error: File not found: {file_path}]"
    except PermissionError:
        return f"[Error: Permission denied: {file_path}]"
    except Exception as e:
        return f"[Error reading file: {type(e).__name__}: {e}]"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating directories if needed.

    Args:
        file_path: Path to the file to write (relative to project root or absolute).
        content: The content to write to the file.

    Returns:
        Success message or error message.
    """
    try:
        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[Successfully wrote to: {file_path}]"
    except PermissionError:
        return f"[Error: Permission denied: {file_path}]"
    except Exception as e:
        return f"[Error writing file: {type(e).__name__}: {e}]"


@tool
def list_directory(dir_path: str) -> str:
    """List files and directories in a path.

    Args:
        dir_path: Path to the directory to list.

    Returns:
        Listing of files and directories, or an error message.
    """
    try:
        entries = os.listdir(dir_path)
        if not entries:
            return f"[Directory is empty: {dir_path}]"
        return "\n".join(sorted(entries))
    except FileNotFoundError:
        return f"[Error: Directory not found: {dir_path}]"
    except PermissionError:
        return f"[Error: Permission denied: {dir_path}]"
    except Exception as e:
        return f"[Error listing directory: {type(e).__name__}: {e}]"


def get_llm() -> ChatOpenAI:
    """Create the LLM instance configured for DeepInfra."""
    return ChatOpenAI(
        model=os.environ.get("DEEPINFRA_MODEL", "deepseek-ai/DeepSeek-V3"),
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
        temperature=0.3,  # Lower temperature for more precise implementation
        max_tokens=8192,  # Larger context for code generation
    )


def get_impl_tools() -> list:
    """Get tools for the implementation agent."""
    return [execute_code, read_file, write_file, list_directory]


def create_impl_agent(checkpointer=None):
    """Create the implementation agent.

    Args:
        checkpointer: State checkpointer for conversation persistence.
            Defaults to MemorySaver.

    Returns:
        The compiled LangGraph agent.
    """
    llm = get_llm()
    tools = get_impl_tools()

    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
    )

    return agent


def _format_task_prompt(task: TaskDefinition, project_root: str) -> str:
    """Format the task definition into a prompt for the agent.

    Args:
        task: The task definition to format.
        project_root: The root directory of the project.

    Returns:
        Formatted prompt string.
    """
    prompt_parts = [
        f"# Task {task.number}: {task.title}",
        "",
        f"**Name**: {task.name}",
        "",
        "## Deliverables",
    ]

    for deliverable in task.deliverables:
        prompt_parts.append(f"- {deliverable}")

    prompt_parts.extend([
        "",
        "## Acceptance Criteria",
    ])

    for criterion in task.acceptance_criteria:
        prompt_parts.append(f"- {criterion}")

    if task.files_to_modify:
        prompt_parts.extend([
            "",
            "## Files to Create/Modify",
        ])
        for file_path in task.files_to_modify:
            prompt_parts.append(f"- {file_path}")

    if task.dependencies:
        prompt_parts.extend([
            "",
            "## Dependencies",
            f"This task depends on: {', '.join(task.dependencies)}",
        ])

    prompt_parts.extend([
        "",
        f"## Project Root",
        f"All file paths should be relative to: {project_root}",
        "",
        "Please implement this task. Read any existing files first, then create or modify files as needed to meet all acceptance criteria.",
    ])

    return "\n".join(prompt_parts)


async def run_implementation(task: TaskDefinition, project_root: str) -> dict:
    """Run the implementation agent on a task.

    Args:
        task: The task definition to implement.
        project_root: The root directory of the project.

    Returns:
        A dictionary with:
            - success: bool indicating whether implementation succeeded
            - files_modified: list of files that were modified
            - output: string containing the agent's response
    """
    agent = create_impl_agent()

    # Build the prompt
    task_prompt = _format_task_prompt(task, project_root)

    # Create the messages
    messages = [
        SystemMessage(content=IMPL_SYSTEM_PROMPT),
        HumanMessage(content=task_prompt),
    ]

    config = {"configurable": {"thread_id": f"task-{task.number}"}}

    try:
        result = await agent.ainvoke({"messages": messages}, config=config)

        # Extract the response
        ai_responses = []
        files_modified = []

        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                ai_responses.append(msg.content)

        # Parse tool calls to find file modifications
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == "write_file":
                        args = tool_call.get("args", {})
                        file_path = args.get("file_path")
                        if file_path and file_path not in files_modified:
                            files_modified.append(file_path)

        output = "\n\n".join(ai_responses)

        # Determine success - basic heuristic: check if we have output and no errors
        success = bool(output) and "[Error" not in output

        return {
            "success": success,
            "files_modified": files_modified,
            "output": output,
        }

    except Exception as e:
        return {
            "success": False,
            "files_modified": [],
            "output": f"Implementation failed with error: {type(e).__name__}: {e}",
        }
