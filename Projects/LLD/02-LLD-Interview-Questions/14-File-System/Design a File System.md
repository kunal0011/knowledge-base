---
date: "2026-04-06"
type: lld-question
difficulty: medium
status: active
tags: [lld, interview-prep, file-system, composite-pattern]
---

# Design a File System

## 1. Problem Statement
Design an in-memory file system with directories, files, creation, deletion, search, and size calculation.

## 2. Class Design

```mermaid
classDiagram
    class FileSystemEntry {
        <<abstract>>
        #String name
        +getSize()* long
        +getPath() String
    }
    class File {
        -bytes content
        +read() bytes
        +write(data)
    }
    class Directory {
        -Map~String,FileSystemEntry~ children
        +addEntry(entry)
        +removeEntry(name)
        +list() List
        +find(name) List
    }

    FileSystemEntry <|-- File
    FileSystemEntry <|-- Directory
    Directory o--> FileSystemEntry
```

### Sequence Flow

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Root as Directory ("/")
    participant Sub as Directory ("etc")
    participant File as File ("hosts", 100B)

    Client->>Root: mkdir("etc")
    Root->>Sub: create child Directory("etc")
    Client->>Sub: createFile("hosts", 100B)
    Sub->>File: instantiate File("hosts")
    Client->>Root: getSize()
    Root->>Sub: getSize() (Recursive Composite)
    Sub->>File: getSize()
    File-->>Sub: 100 Bytes
    Sub-->>Root: 100 Bytes
    Root-->>Client: Total Size: 100 Bytes
```


## 3. Key Implementation (Python)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class FileSystemEntry(ABC):
    def __init__(self, name: str, parent=None):
        self.name = name
        self.parent = parent

    @abstractmethod
    def get_size(self) -> int: pass

    def get_path(self) -> str:
        if self.parent is None:
            return '/' + self.name
        return self.parent.get_path() + '/' + self.name

class File(FileSystemEntry):
    def __init__(self, name: str, parent=None):
        super().__init__(name, parent)
        self._content = b''

    def get_size(self) -> int:
        return len(self._content)

    def write(self, data: bytes):
        self._content = data

    def read(self) -> bytes:
        return self._content

class Directory(FileSystemEntry):
    def __init__(self, name: str, parent=None):
        super().__init__(name, parent)
        self._children: Dict[str, FileSystemEntry] = {}

    def get_size(self) -> int:
        return sum(c.get_size() for c in self._children.values())

    def add(self, entry: FileSystemEntry):
        entry.parent = self
        self._children[entry.name] = entry

    def remove(self, name: str):
        self._children.pop(name, None)

    def list_entries(self) -> List[str]:
        return list(self._children.keys())

    def find(self, name: str) -> List[FileSystemEntry]:
        results = []
        for child in self._children.values():
            if child.name == name:
                results.append(child)
            if isinstance(child, Directory):
                results.extend(child.find(name))
        return results

class FileSystem:
    def __init__(self):
        self.root = Directory('')

    def mkdir(self, path: str) -> Directory:
        parts = [p for p in path.split('/') if p]
        current = self.root
        for part in parts:
            entry = current._children.get(part)
            if entry is None:
                new_dir = Directory(part)
                current.add(new_dir)
                current = new_dir
            elif isinstance(entry, Directory):
                current = entry
        return current

    def touch(self, path: str) -> File:
        parts = [p for p in path.split('/') if p]
        directory = self.mkdir('/'.join(parts[:-1]))
        f = File(parts[-1])
        directory.add(f)
        return f
```

### Java

```java
package com.lld.filesystem;

import java.util.*;

interface FileSystemNode {
    String getName();
    int getSize();
    boolean isDirectory();
}

class FileNode implements FileSystemNode {
    private final String name;
    private byte[] content;

    public FileNode(String name, byte[] content) {
        this.name = name;
        this.content = content != null ? content : new byte[0];
    }
    @Override public String getName() { return name; }
    @Override public int getSize() { return content.length; }
    @Override public boolean isDirectory() { return false; }
}

class DirectoryNode implements FileSystemNode {
    private final String name;
    private final Map<String, FileSystemNode> children = new HashMap<>();

    public DirectoryNode(String name) { this.name = name; }
    @Override public String getName() { return name; }

    @Override
    public int getSize() {
        int total = 0;
        for (FileSystemNode node : children.values()) {
            total += node.getSize();
        }
        return total;
    }

    @Override public boolean isDirectory() { return true; }

    public void addNode(FileSystemNode node) { children.put(node.getName(), node); }
    public FileSystemNode getChild(String name) { return children.get(name); }
}

public class FileSystem {
    private final DirectoryNode root = new DirectoryNode("/");

    public DirectoryNode getRoot() { return root; }

    public int calculateTotalSize() {
        return root.getSize();
    }
}
```


## 4. Patterns: **Composite** (files + directories) | **Iterator** (traversal) | **Visitor** (operations)


---

## Thread Safety Considerations

| Concern | Solution |
|---|---|
| Concurrent file tree traversal | Immutable nodes or ReadWriteLock on directory child map allows concurrent readers |
| Node addition race | Per-directory locks avoid tree restructuring race conditions during path creation |

## Extensibility & SOLID Principles

| Principle | Architectural Implementation |
|---|---|
| **S** — Single Responsibility | `FileNode` stores content bytes; `DirectoryNode` maintains hierarchical child links |
| **O** — Open/Closed | New node types (Symlink, Pipe, BlockDevice) implement `FileSystemNode` without changing callers |
| **L** — Liskov Substitution | `FileNode` and `DirectoryNode` are completely interchangeable via the `FileSystemNode` interface |

---

## 5. Follow-ups
- **Permissions?** `Permission` class with rwx bits per user/group.
- **Symlinks?** `SymLink` pointing to another entry.
- **Disk-backed?** Proxy pattern for lazy content loading.

---

**Related:** [[12 - Composite Pattern]] | [[11 - Iterator Pattern]] | [[14 - Proxy Pattern]]
