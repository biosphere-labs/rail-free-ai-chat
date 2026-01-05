from pathlib import Path
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pool import TaskResult, PoolStatus


def update_task_status(epic_folder: str, task_num: int, new_status: str) -> None:
    """Update status in task markdown file frontmatter."""
    task_file = Path(epic_folder) / f"{task_num:03d}.md"
    if not task_file.exists():
        return

    content = task_file.read_text()
    lines = content.split("\n")

    # Find and update status in frontmatter
    in_frontmatter = False
    for i, line in enumerate(lines):
        if line.strip() == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter and line.startswith("status:"):
            lines[i] = f"status: {new_status}"
        if in_frontmatter and line.startswith("updated:"):
            lines[i] = f"updated: {datetime.now(timezone.utc).isoformat()}"

    task_file.write_text("\n".join(lines))


def update_execution_status(epic_folder: str, status: "PoolStatus") -> None:
    """Update or create execution-status.md with current progress."""
    status_file = Path(epic_folder) / "execution-status.md"

    total = len(status.completed) + len(status.failed) + len(status.in_progress)
    completed = len(status.completed)
    failed = len(status.failed)
    in_progress = len(status.in_progress)

    content = f"""---
updated: {datetime.now(timezone.utc).isoformat()}
total_tasks: {total}
completed: {completed}
failed: {failed}
in_progress: {in_progress}
---

# Execution Status

## Summary
- **Completed**: {completed}/{total}
- **Failed**: {failed}
- **In Progress**: {in_progress}

## Task Results

"""

    for task_num in sorted(status.results.keys()):
        result = status.results[task_num]
        icon = "✅" if result.success else "❌"
        content += f"### Task {task_num:03d} {icon}\n"
        if result.files_modified:
            content += f"Files: {', '.join(result.files_modified)}\n"
        if result.error:
            content += f"Error: {result.error}\n"
        content += "\n"

    status_file.write_text(content)


async def on_task_complete_callback(epic_folder: str):
    """Create callback for updating status on task completion."""
    async def callback(result: "TaskResult") -> None:
        new_status = "completed" if result.success else "failed"
        update_task_status(epic_folder, result.task_num, new_status)
    return callback
