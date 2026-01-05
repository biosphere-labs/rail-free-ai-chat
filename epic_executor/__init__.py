"""Epic executor - parallel task execution from epic specifications."""

from .parser import TaskDefinition, parse_epic_folder
from .scheduler import create_execution_plan, ExecutionPlan, get_ready_tasks
from .pool import run_pool, PoolStatus, TaskResult
from .impl_agent import run_implementation, create_impl_agent
from .verify_agent import run_verification, create_verify_agent
from .executor import execute_epic, dry_run
from .status import update_execution_status

__all__ = [
    "TaskDefinition",
    "parse_epic_folder",
    "create_execution_plan",
    "ExecutionPlan",
    "get_ready_tasks",
    "run_pool",
    "PoolStatus",
    "TaskResult",
    "run_implementation",
    "create_impl_agent",
    "run_verification",
    "create_verify_agent",
    "execute_epic",
    "dry_run",
    "update_execution_status",
]
