---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 206: Reverse Linked List"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 206: Reverse Linked List

---

### Problem Statement

Given the `head` of a singly linked list, reverse the list, and return the reversed list.

---

### Key Observation

* Maintain three pointers: `prev`, `curr`, and `next_temp`.
* Invert the `curr.next = prev` pointer on each iteration while advancing forward.

---

### Core Technique: Three-Pointer Iterative Reversal

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev
```

---

### Worked-Out Example

```python
1 -> 2 -> 3 -> None
curr=1: 1.next = None, prev = 1, curr = 2
curr=2: 2.next = 1, prev = 2, curr = 3
curr=3: 3.next = 2, prev = 3, curr = None
return prev = 3 -> 2 -> 1 -> None
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Store `curr.next` before overwriting the pointer to preserve list traversal.