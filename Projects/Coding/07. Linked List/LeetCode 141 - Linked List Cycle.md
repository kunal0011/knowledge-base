---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 141: Linked List Cycle"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 141: Linked List Cycle

---

### Problem Statement

Given `head` of a linked list, determine if the linked list has a cycle in it.

---

### Key Observation

* Use Floyd's Cycle-Finding Algorithm (Fast and Slow pointers).
* `slow` moves 1 step, `fast` moves 2 steps.
* If a cycle exists, `fast` will catch up to `slow`.

---

### Core Technique: Floyd's Tortoise and Hare Algorithm

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

---

### Worked-Out Example

```python
3 -> 2 -> 0 -> -4 -> (loops back to 2)
slow=3, fast=3
step 1: slow=2, fast=0
step 2: slow=0, fast=2
step 3: slow=-4, fast=-4 (slow == fast!) -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Fast (2x) and Slow (1x) pointer convergence guarantees cycle detection without O(n) memory.