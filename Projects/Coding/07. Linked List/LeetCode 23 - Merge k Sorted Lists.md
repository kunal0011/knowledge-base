---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 23: Merge k Sorted Lists"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 23: Merge k Sorted Lists

---

### Problem Statement

You are given an array of `k` linked-lists, each sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

---

### Key Observation

* Pairwise merge lists using Divide and Conquer (similar to Merge Sort).
* Merging in pairs reduces `k` lists to `k/2`, then `k/4`... taking `log k` rounds.

---

### Core Technique: Divide and Conquer List Merging

---

### Python 3 Solution (with typing)

```python
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
            
        while len(lists) > 1:
            merged_lists = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                merged_lists.append(self.merge2Lists(l1, l2))
            lists = merged_lists
        return lists[0]
        
    def merge2Lists(self, l1, l2):
        dummy = ListNode()
        tail = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 or l2
        return dummy.next
```

---

### Worked-Out Example

```python
lists = [[1->4->5], [1->3->4], [2->6]]
Round 1: merge list 0 and 1 -> [1->1->3->4->4->5], keep list 2 [2->6]
Round 2: merge -> [1->1->2->3->4->4->5->6]
```

---

### Complexity Analysis

* **Time Complexity:** `O(N log k) where N is total nodes across all k lists`
* **Space Complexity:** `O(1) auxiliary`

---

### Takeaway Pattern

Divide and Conquer reduces sequential O(k \* N) merges to optimal O(N log k).