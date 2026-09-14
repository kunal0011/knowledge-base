---
date: "2026-04-06"
type: lld-question
difficulty: medium
tags: [lld, interview-prep, todo-list, command-pattern]
---

# Design a Todo List Application

## 1. Problem Statement
Design a task management app with CRUD, priorities, due dates, labels, filtering, and undo support.

## 2. Key Implementation (Python)

```python
from enum import Enum
from datetime import datetime, date
from typing import List, Dict, Optional, Set
import uuid

class Priority(Enum):
    LOW = 1; MEDIUM = 2; HIGH = 3; URGENT = 4

class TaskStatus(Enum):
    TODO = "TODO"; IN_PROGRESS = "IN_PROGRESS"; DONE = "DONE"

class Task:
    def __init__(self, title: str, description: str = "",
                 priority: Priority = Priority.MEDIUM, due_date: date = None):
        self.task_id = str(uuid.uuid4())[:8]
        self.title = title; self.description = description
        self.priority = priority; self.due_date = due_date
        self.status = TaskStatus.TODO; self.labels: Set[str] = set()
        self.created_at = datetime.now(); self.completed_at = None

    def complete(self):
        self.status = TaskStatus.DONE
        self.completed_at = datetime.now()

class TodoList:
    def __init__(self, name: str, owner: str):
        self.name = name; self.owner = owner
        self.tasks: Dict[str, Task] = {}

    def add_task(self, task: Task) -> str:
        self.tasks[task.task_id] = task; return task.task_id

    def remove_task(self, task_id: str): self.tasks.pop(task_id, None)

    def complete_task(self, task_id: str):
        if task_id in self.tasks: self.tasks[task_id].complete()

    def get_by_priority(self, priority: Priority) -> List[Task]:
        return [t for t in self.tasks.values() if t.priority == priority]

    def get_by_label(self, label: str) -> List[Task]:
        return [t for t in self.tasks.values() if label in t.labels]

    def get_overdue(self) -> List[Task]:
        today = date.today()
        return [t for t in self.tasks.values()
                if t.due_date and t.due_date < today and t.status != TaskStatus.DONE]

    def get_sorted(self, by: str = "priority") -> List[Task]:
        if by == "priority":
            return sorted(self.tasks.values(), key=lambda t: t.priority.value, reverse=True)
        elif by == "due_date":
            return sorted(self.tasks.values(), key=lambda t: t.due_date or date.max)
        return list(self.tasks.values())
```

## 3. Patterns: **Command** (undo/redo) | **Observer** (due date reminders) | **Strategy** (sorting/filtering)
## 4. Follow-ups: **Subtasks?** Composite pattern | **Collaboration?** Shared lists with permissions | **Recurring tasks?** Template that generates instances.

---
**Related:** [[07 - Command Pattern]] | [[12 - Composite Pattern]]
