---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 25: Reverse Nodes in k-Group"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 25: Reverse Nodes in k-Group

---

### Problem Statement

Given the head of a linked list, reverse the nodes of the list `k` at a time, and return the modified list. If number of nodes is not a multiple of `k`, leave remaining nodes as-is.

---

### Key Observation

* Find the k-th node from current segment. If fewer than `k` nodes remain, stop.
* Reverse exactly `k` nodes iteratively and reconnect `groupPrev.next` to new group head.
* Update `groupPrev` to point to the end of the newly reversed group.

---

### Core Technique: Iterative Segment Pointer Reversal with Group Anchors

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy
        
        while True:
            kth = self.get_kth(group_prev, k)
            if not kth:
                break
            group_next = kth.next
            
            # Reverse group
            prev, curr = kth.next, group_prev.next
            while curr != group_next:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp
                
            tmp = group_prev.next
            group_prev.next = kth
            group_prev = tmp
            
        return dummy.next

    def get_kth(self, curr, k):
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
```

---

### Worked-Out Example

```python
list = 1 -> 2 -> 3 -> 4 -> 5, k = 2
Group 1 [1, 2]: reversed -> 2 -> 1 -> 3 -> 4 -> 5
Group 2 [3, 4]: reversed -> 2 -> 1 -> 4 -> 3 -> 5
Group 3 [5]: length < 2 -> unchanged
Result = 2 -> 1 -> 4 -> 3 -> 5
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Anchor each group with `groupPrev` and `groupNext` pointers to reverse sub-lists cleanly in place.