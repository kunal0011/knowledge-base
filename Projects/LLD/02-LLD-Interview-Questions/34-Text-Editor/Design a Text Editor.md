---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, text-editor, command-pattern]
---

# Design a Text Editor (Undo/Redo)

## 1. Problem Statement
Design a text editor with insert, delete, cursor movement, and undo/redo functionality using the Command pattern.

## 2. Key Implementation (Python)

```python
from abc import ABC, abstractmethod
from typing import List

class Command(ABC):
    @abstractmethod
    def execute(self): pass
    @abstractmethod
    def undo(self): pass

class InsertCommand(Command):
    def __init__(self, editor: 'TextEditor', position: int, text: str):
        self._editor = editor
        self._position = position
        self._text = text

    def execute(self):
        self._editor.buffer = (self._editor.buffer[:self._position] +
                                self._text +
                                self._editor.buffer[self._position:])
        self._editor.cursor = self._position + len(self._text)

    def undo(self):
        self._editor.buffer = (self._editor.buffer[:self._position] +
                                self._editor.buffer[self._position + len(self._text):])
        self._editor.cursor = self._position

class DeleteCommand(Command):
    def __init__(self, editor: 'TextEditor', position: int, length: int):
        self._editor = editor
        self._position = position
        self._length = length
        self._deleted_text = ""

    def execute(self):
        self._deleted_text = self._editor.buffer[self._position:self._position + self._length]
        self._editor.buffer = (self._editor.buffer[:self._position] +
                                self._editor.buffer[self._position + self._length:])
        self._editor.cursor = self._position

    def undo(self):
        self._editor.buffer = (self._editor.buffer[:self._position] +
                                self._deleted_text +
                                self._editor.buffer[self._position:])
        self._editor.cursor = self._position + len(self._deleted_text)

class TextEditor:
    def __init__(self):
        self.buffer = ""
        self.cursor = 0
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    def insert(self, text: str):
        cmd = InsertCommand(self, self.cursor, text)
        cmd.execute()
        self._undo_stack.append(cmd)
        self._redo_stack.clear()

    def delete(self, length: int = 1):
        if self.cursor > 0:
            pos = self.cursor - length
            cmd = DeleteCommand(self, pos, length)
            cmd.execute()
            self._undo_stack.append(cmd)
            self._redo_stack.clear()

    def undo(self):
        if self._undo_stack:
            cmd = self._undo_stack.pop()
            cmd.undo()
            self._redo_stack.append(cmd)
            print(f"↩️ Undo: '{self.buffer}'")

    def redo(self):
        if self._redo_stack:
            cmd = self._redo_stack.pop()
            cmd.execute()
            self._undo_stack.append(cmd)
            print(f"↪️ Redo: '{self.buffer}'")

    def __str__(self):
        return f"'{self.buffer}' (cursor: {self.cursor})"

if __name__ == "__main__":
    editor = TextEditor()
    editor.insert("Hello ")
    editor.insert("World")
    print(editor)  # 'Hello World'
    editor.undo()
    print(editor)  # 'Hello '
    editor.undo()
    print(editor)  # ''
    editor.redo()
    print(editor)  # 'Hello '
```

## 3. Patterns: **Command** (undo/redo) | **Memento** (snapshot state)

## 4. Follow-ups
- **Find & replace?** Composite command with delete + insert.
- **Clipboard?** Memento pattern for copy/paste buffer.
- **Collaborative editing?** Operational Transformation or CRDT.

---

**Related:** [[07 - Command Pattern]] | [[02 - Observer Pattern]]
