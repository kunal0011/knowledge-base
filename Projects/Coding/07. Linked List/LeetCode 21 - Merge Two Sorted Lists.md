---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 21: Merge Two Sorted Lists"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 21: Merge Two Sorted Lists

---

### Problem Statement

Merge two sorted linked lists into one sorted list and return its head.

---

### Key Observation

* Use a **dummy head node** to avoid edge-case checks for the new list head.
* Compare heads of both lists and advance the smaller node pointer.

---

### Core Technique: Dummy Head Node Two-Pointer Merge

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
            
        tail.next = list1 if list1 else list2
        return dummy.next
```

---

### Worked-Out Example

```python
list1 = 1 -> 2 -> 4, list2 = 1 -> 3 -> 4
dummy -> 1 -> 1 -> 2 -> 3 -> 4 -> 4
return dummy.next (head of merged list)
```

---

### Complexity Analysis

* **Time Complexity:** `O(n + m)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Always create a dummy node `dummy = ListNode()` when building a new linked list from scratch.