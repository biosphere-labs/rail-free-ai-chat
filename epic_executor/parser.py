import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TaskDefinition:
    task_number: int
    name: str
    title: str | None = None
    status: str = "open"
    deliverables: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    files_to_create: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)
    frontmatter: dict = field(default_factory=dict)

    @property
    def number(self) -> int:
        return self.task_number


def parse_task_file(path: Path) -> TaskDefinition:
    content = path.read_text()

    task_number = int(path.stem)

    frontmatter = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1]) or {}
            body = parts[2]

    name = frontmatter.get("name", "")
    title = frontmatter.get("title")
    status = frontmatter.get("status", "open")

    deliverables = extract_section(body, "Deliverables")
    acceptance_criteria = extract_checklist(body, "Acceptance Criteria")
    files_to_create = extract_file_list(body, "Files to Create")
    files_to_modify = extract_file_list(body, "Files to Modify")
    dependencies = extract_dependencies(body, frontmatter, task_number)

    return TaskDefinition(
        task_number=task_number,
        name=name,
        title=title,
        status=status,
        deliverables=deliverables,
        acceptance_criteria=acceptance_criteria,
        files_to_create=files_to_create,
        files_to_modify=files_to_modify,
        dependencies=dependencies,
        frontmatter=frontmatter,
    )


def extract_section(body: str, section_name: str) -> str:
    pattern = rf"^##\s+{re.escape(section_name)}\s*\n(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_checklist(body: str, section_name: str) -> list[str]:
    section = extract_section(body, section_name)
    items = []
    for line in section.split("\n"):
        match = re.match(r"^\s*-\s*\[[ xX]\]\s*(.+)$", line)
        if match:
            items.append(match.group(1).strip())
    return items


def extract_file_list(body: str, section_name: str) -> list[str]:
    section = extract_section(body, section_name)
    files = []
    for line in section.split("\n"):
        match = re.match(r"^\s*-\s*`?([^`\n]+)`?\s*$", line)
        if match:
            files.append(match.group(1).strip())
    return files


def extract_dependencies(body: str, frontmatter: dict, self_task_num: int) -> list[int]:
    """Extract dependencies from frontmatter depends_on field only.

    Note: We intentionally don't scan body text for "Task X" references
    because those are often forward references or notes, not actual dependencies.
    """
    deps = set()

    if "depends_on" in frontmatter:
        depends_on = frontmatter["depends_on"]
        if isinstance(depends_on, list):
            for d in depends_on:
                if isinstance(d, int):
                    deps.add(d)
                elif isinstance(d, str) and d.isdigit():
                    deps.add(int(d))

    # Remove self-reference just in case
    deps.discard(self_task_num)

    return sorted(deps)


def parse_epic_folder(path: str) -> list[TaskDefinition]:
    folder = Path(path)
    tasks = []

    for md_file in folder.glob("[0-9]*.md"):
        if md_file.stem.isdigit():
            tasks.append(parse_task_file(md_file))

    return sorted(tasks, key=lambda t: t.task_number)
