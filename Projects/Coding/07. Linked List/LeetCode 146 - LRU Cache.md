---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 146: LRU Cache"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 146: LRU Cache

---

### Problem Statement

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache with `O(1)` average time complexity for `get` and `put`.

---

### Key Observation

* A Hash Map provides `O(1)` key-to-node lookup.
* A Doubly Linked List provides `O(1)` node removal and insertion at head/tail.
* Use sentinel head and tail dummy nodes to simplify boundary operations.

---

### Core Technique: Doubly Linked List + Hash Map

---

### Python 3 Solution (with typing)

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}  # key -> Node
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def _insert_at_head(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._insert_at_head(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self.cache[key] = node
        self._insert_at_head(node)
        
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

---

### Worked-Out Example

```python
LRU capacity = 2
put(1, 1) -> head <-> [1:1] <-> tail
put(2, 2) -> head <-> [2:2] <-> [1:1] <-> tail
get(1) -> returns 1, moves [1:1] to front -> head <-> [1:1] <-> [2:2] <-> tail
put(3, 3) -> evicts LRU [2:2] -> head <-> [3:3] <-> [1:1] <-> tail
```

---

### Complexity Analysis

* **Time Complexity:** `O(1) for both get and put`
* **Space Complexity:** `O(capacity)`

---

### Takeaway Pattern

Combine Doubly Linked List for O(1) order re-linking with Hash Map for O(1) direct access.