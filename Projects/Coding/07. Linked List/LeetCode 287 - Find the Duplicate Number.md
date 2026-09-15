---
date: "2026-08-29"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 287: Find the Duplicate Number"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 287: Find the Duplicate Number

---

### Problem Statement

Given an array of integers `nums` containing `n + 1` integers where each integer is in range `[1, n]` inclusive. Find the duplicate number without modifying array and in `O(1)` extra space.

---

### Key Observation

* Since values are in `[1, n]`, treat values as pointers to indices: `index -> nums[index]`.
* A duplicate value means multiple indices point to the same next node, creating a cycle.
* Use Floyd's Tortoise and Hare algorithm to detect the cycle, then find the entry point.

---

### Core Technique: Array as Linked List (Floyd's Cycle Entry Detection)

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Phase 1: Detect cycle
        slow = nums[0]
        fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
                
        # Phase 2: Find entrance to cycle
        ptr1 = nums[0]
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
            
        return ptr1
```

---

### Worked-Out Example

```python
nums = [1, 3, 4, 2, 2]
0 -> 1 -> 3 -> 2 -> 4 -> 2 (cycle at 2)
Phase 1: slow and fast meet at 4
Phase 2: ptr1=1, ptr2=4 -> ptr1=nums[1]=3, ptr2=nums[4]=2 -> ptr1=nums[3]=2, ptr2=nums[2]=2
Duplicate is 2!
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Map `nums[i]` to `next` pointer to apply linked list cycle detection algorithms to arrays.