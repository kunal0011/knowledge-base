---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 143: Reorder List"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 143: Reorder List

---

### Problem Statement

You are given the head of a singly linked list `L0 -> L1 -> ... -> Ln-1 -> Ln`. Reorder it to `L0 -> Ln -> L1 -> Ln-1 -> L2 -> Ln-2...` in-place.

---

### Key Observation

* Step 1: Find the middle of the linked list using Fast & Slow pointers.
* Step 2: Reverse the second half of the list.
* Step 3: Merge/interleave the first half and the reversed second half.

---

### Core Technique: Find Middle + Reverse Second Half + Interleave

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return
            
        # 1. Find middle
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        # 2. Reverse second half
        second = slow.next
        slow.next = None
        prev = None
        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp
            
        # 3. Interleave two halves
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2
```

---

### Worked-Out Example

```python
1 -> 2 -> 3 -> 4 -> 5
Middle split: [1 -> 2 -> 3] and [4 -> 5]
Reverse second: [5 -> 4]
Interleave: 1 -> 5 -> 2 -> 4 -> 3
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1) in-place`

---

### Takeaway Pattern

Complex linked list reorderings often decompose into: find middle -> reverse half -> interleave.