import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Awaitable
from pathlib import Path

if TYPE_CHECKING:
    from .parser import TaskDefinition
    from .scheduler import ExecutionPlan


@dataclass
class TaskResult:
    task_num: int
    success: bool
    impl_output: str = ""
    verify_output: str = ""
    files_modified: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class PoolStatus:
    completed: set[int] = field(default_factory=set)
    in_progress: set[int] = field(default_factory=set)
    failed: set[int] = field(default_factory=set)
    results: dict[int, TaskResult] = field(default_factory=dict)


async def execute_task(
    task: "TaskDefinition",
    project_root: str,
    impl_fn: Callable,
    verify_fn: Callable,
) -> TaskResult:
    """Execute a single task: implement then verify."""
    try:
        # Run implementation
        impl_result = await impl_fn(task, project_root)

        if not impl_result.get("success", False):
            return TaskResult(
                task_num=task.number,
                success=False,
                impl_output=impl_result.get("output", ""),
                error="Implementation failed",
            )

        # Run verification
        verify_result = await verify_fn(task, project_root, impl_result)

        return TaskResult(
            task_num=task.number,
            success=verify_result.get("passed", False),
            impl_output=impl_result.get("output", ""),
            verify_output=verify_result.get("test_output", ""),
            files_modified=impl_result.get("files_modified", []),
            error="" if verify_result.get("passed") else "Verification failed",
        )

    except Exception as e:
        return TaskResult(
            task_num=task.number,
            success=False,
            error=str(e),
        )


async def run_pool(
    tasks: list["TaskDefinition"],
    plan: "ExecutionPlan",
    project_root: str,
    impl_fn: Callable,
    verify_fn: Callable,
    max_concurrent: int = 4,
    on_task_complete: Callable[[TaskResult], Awaitable[None]] = None,
) -> PoolStatus:
    """Run tasks respecting dependencies with concurrent execution."""
    from .scheduler import get_ready_tasks

    task_map = {t.number: t for t in tasks}
    status = PoolStatus()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task: "TaskDefinition") -> TaskResult:
        async with semaphore:
            return await execute_task(task, project_root, impl_fn, verify_fn)

    pending_futures: dict[int, asyncio.Task] = {}

    while len(status.completed) + len(status.failed) < len(tasks):
        # Find ready tasks
        ready = get_ready_tasks(plan, status.completed, status.in_progress)

        # Start new tasks
        for task_num in ready:
            if task_num not in pending_futures:
                task = task_map[task_num]
                status.in_progress.add(task_num)
                future = asyncio.create_task(run_with_semaphore(task))
                pending_futures[task_num] = future

        if not pending_futures:
            break  # No more work

        # Wait for at least one to complete
        done, _ = await asyncio.wait(
            pending_futures.values(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Process completed
        for future in done:
            result = future.result()
            task_num = result.task_num

            status.in_progress.discard(task_num)
            status.results[task_num] = result

            if result.success:
                status.completed.add(task_num)
            else:
                status.failed.add(task_num)

            if on_task_complete:
                await on_task_complete(result)

            del pending_futures[task_num]

    return status
