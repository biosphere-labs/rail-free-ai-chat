"""Main epic executor - orchestrates the full pipeline."""

import asyncio
from pathlib import Path
from typing import Callable

from .parser import parse_epic_folder, TaskDefinition
from .scheduler import create_execution_plan, ExecutionPlan
from .pool import run_pool, PoolStatus, TaskResult
from .status import update_execution_status, on_task_complete_callback
from .impl_agent import run_implementation
from .verify_agent import run_verification


async def execute_epic(
    epic_folder: str,
    project_root: str,
    max_concurrent: int = 4,
    on_progress: Callable[[str], None] = None,
) -> PoolStatus:
    """Execute all tasks in an epic folder.

    Args:
        epic_folder: Path to folder containing numbered task .md files
        project_root: Root directory of the project to implement tasks in
        max_concurrent: Maximum number of parallel agents
        on_progress: Optional callback for progress updates

    Returns:
        PoolStatus with results for all tasks
    """
    def log(msg: str):
        if on_progress:
            on_progress(msg)
        print(msg)

    # Parse all tasks
    log(f"Parsing epic folder: {epic_folder}")
    tasks = parse_epic_folder(epic_folder)
    log(f"Found {len(tasks)} tasks")

    if not tasks:
        log("No tasks found!")
        return PoolStatus()

    # Build execution plan
    log("Building dependency graph and execution plan...")
    plan = create_execution_plan(tasks)
    log(f"Execution levels: {plan.levels}")

    # Create status update callback
    status_callback = await on_task_complete_callback(epic_folder)

    async def progress_callback(result: TaskResult):
        await status_callback(result)
        status = "✅" if result.success else "❌"
        log(f"Task {result.task_num:03d} {status}")

    # Run the pool
    log(f"Starting execution with {max_concurrent} concurrent agents...")
    status = await run_pool(
        tasks=tasks,
        plan=plan,
        project_root=project_root,
        impl_fn=run_implementation,
        verify_fn=run_verification,
        max_concurrent=max_concurrent,
        on_task_complete=progress_callback,
    )

    # Update final status
    update_execution_status(epic_folder, status)

    # Summary
    log(f"\nExecution complete:")
    log(f"  Completed: {len(status.completed)}/{len(tasks)}")
    log(f"  Failed: {len(status.failed)}")

    return status


async def dry_run(epic_folder: str) -> ExecutionPlan:
    """Parse and plan without executing - useful for preview."""
    tasks = parse_epic_folder(epic_folder)
    plan = create_execution_plan(tasks)

    print(f"Epic: {epic_folder}")
    print(f"Tasks: {len(tasks)}")
    print(f"\nExecution Plan (tasks at same level run in parallel):")

    for i, level in enumerate(plan.levels):
        task_info = []
        task_map = {t.number: t for t in tasks}
        for num in level:
            task = task_map.get(num)
            name = task.name if task else "unknown"
            task_info.append(f"{num:03d} ({name})")
        print(f"  Level {i}: {', '.join(task_info)}")

    print(f"\nDependency Map:")
    for num, deps in plan.dependency_map.items():
        if deps:
            print(f"  Task {num:03d} depends on: {deps}")

    return plan
