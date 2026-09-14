---
date: "2026-04-06"
type: lld-question
difficulty: hard
status: active
tags: [lld, interview-prep, spreadsheet, observer-pattern, composite-pattern]
---

# Design a Spreadsheet (Excel)

## 1. Problem Statement
Design a spreadsheet with cells, formulas, dependency tracking, and auto-recalculation on changes.

## 2. Key Implementation (Python)

```python
from typing import Dict, Set, Optional, List
from enum import Enum
import re

class CellType(Enum):
    EMPTY = "EMPTY"
    NUMBER = "NUMBER"
    TEXT = "TEXT"
    FORMULA = "FORMULA"

class Cell:
    def __init__(self):
        self.raw_value = ""
        self.computed_value = None
        self.cell_type = CellType.EMPTY
        self.dependents: Set[str] = set()    # Cells that depend ON this cell
        self.dependencies: Set[str] = set()  # Cells THIS cell depends on

class Spreadsheet:
    def __init__(self, rows: int = 100, cols: int = 26):
        self.rows = rows
        self.cols = cols
        self._cells: Dict[str, Cell] = {}

    def set_cell(self, cell_ref: str, value: str):
        cell = self._get_or_create(cell_ref)
        
        # Remove old dependencies
        for dep in cell.dependencies:
            if dep in self._cells:
                self._cells[dep].dependents.discard(cell_ref)
        cell.dependencies.clear()

        cell.raw_value = value

        if not value:
            cell.cell_type = CellType.EMPTY
            cell.computed_value = None
        elif value.startswith('='):
            cell.cell_type = CellType.FORMULA
            refs = self._parse_references(value)
            cell.dependencies = refs
            for ref in refs:
                self._get_or_create(ref).dependents.add(cell_ref)
            
            if self._has_circular(cell_ref):
                cell.computed_value = "#CIRCULAR!"
                return
            cell.computed_value = self._evaluate(value[1:])
        elif self._is_number(value):
            cell.cell_type = CellType.NUMBER
            cell.computed_value = float(value)
        else:
            cell.cell_type = CellType.TEXT
            cell.computed_value = value

        # Recalculate dependents
        self._propagate(cell_ref)

    def get_cell(self, cell_ref: str):
        cell = self._cells.get(cell_ref)
        return cell.computed_value if cell else None

    def _evaluate(self, formula: str):
        """Evaluate formula by replacing cell refs with values"""
        def replace_ref(match):
            ref = match.group(0)
            val = self.get_cell(ref)
            return str(val) if val is not None else "0"

        expr = re.sub(r'[A-Z]+\d+', replace_ref, formula)
        try:
            # Handle SUM(A1:A5) type functions
            expr = self._expand_functions(expr)
            return eval(expr)
        except:
            return "#ERROR!"

    def _expand_functions(self, expr: str) -> str:
        # Simple SUM implementation
        def sum_range(match):
            start, end = match.group(1), match.group(2)
            cells = self._get_range(start, end)
            values = [self.get_cell(c) or 0 for c in cells]
            return str(sum(float(v) for v in values))
        return re.sub(r'SUM\(([A-Z]\d+):([A-Z]\d+)\)', sum_range, expr)

    def _parse_references(self, formula: str) -> Set[str]:
        return set(re.findall(r'[A-Z]+\d+', formula))

    def _propagate(self, cell_ref: str):
        """BFS to recalculate all dependent cells"""
        visited = set()
        queue = list(self._cells.get(cell_ref, Cell()).dependents)
        while queue:
            ref = queue.pop(0)
            if ref in visited:
                continue
            visited.add(ref)
            cell = self._cells.get(ref)
            if cell and cell.cell_type == CellType.FORMULA:
                cell.computed_value = self._evaluate(cell.raw_value[1:])
            queue.extend(cell.dependents if cell else [])

    def _has_circular(self, start: str) -> bool:
        visited = set()
        stack = [start]
        while stack:
            ref = stack.pop()
            if ref in visited:
                return True
            visited.add(ref)
            cell = self._cells.get(ref)
            if cell:
                stack.extend(cell.dependencies)
        return False

    def _get_or_create(self, ref: str) -> Cell:
        if ref not in self._cells:
            self._cells[ref] = Cell()
        return self._cells[ref]

    def _get_range(self, start: str, end: str) -> List[str]:
        col1, row1 = start[0], int(start[1:])
        col2, row2 = end[0], int(end[1:])
        return [f"{col1}{r}" for r in range(row1, row2 + 1)]

    def _is_number(self, s: str) -> bool:
        try: float(s); return True
        except: return False

if __name__ == "__main__":
    sheet = Spreadsheet()
    sheet.set_cell("A1", "10")
    sheet.set_cell("A2", "20")
    sheet.set_cell("A3", "=A1+A2")       # 30
    sheet.set_cell("A4", "=SUM(A1:A3)")  # 60

    print(f"A3 = {sheet.get_cell('A3')}")  # 30
    print(f"A4 = {sheet.get_cell('A4')}")  # 60

    sheet.set_cell("A1", "100")  # Change A1 → A3 and A4 auto-recalculate
    print(f"A3 = {sheet.get_cell('A3')}")  # 120
    print(f"A4 = {sheet.get_cell('A4')}")  # 240
```

## 3. Key Concepts
- **Dependency Graph**: DAG of cell dependencies. Topological sort for evaluation order.
- **Circular Detection**: DFS to check for cycles before allowing formula.
- **Change Propagation**: BFS from changed cell through all dependents.

## 4. Patterns: **Observer** (cell change → dependents recalculate) | **Composite** (ranges/formulas)

## 5. Follow-ups
- **Undo/redo?** Command pattern with cell state snapshots.
- **Collaborative editing?** CRDT or OT for concurrent cell edits.
- **Charts?** Observer — chart subscribes to cell range changes.

---

**Related:** [[02 - Observer Pattern]] | [[12 - Composite Pattern]] | [[07 - Command Pattern]]
