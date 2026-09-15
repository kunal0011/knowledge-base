---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 143: Reorder List"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 143: Reorder List

Below is a **complete, structured explanation** for **LeetCode 143 – Reorder List**, aligned with how this problem is typically expected to be reasoned about in interviews.

---

## LeetCode 143 — Reorder List

### Problem Statement

You are given the head of a **singly linked list**:

```
L0 → L1 → L2 → … → Ln-1 → Ln
```

Reorder the list to follow this pattern:

```
L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …
```

* You **must modify the list in-place**
* You **cannot change node values**, only pointers

---

## Key Observation

1. The reorder pattern **alternates between the front and the back**.
2. Singly linked lists **cannot traverse backward**, so direct access to the tail is not possible.
3. Therefore, we need a way to:

   * Access nodes from the **end**
   * Interleave them with nodes from the **start**

This naturally suggests **reversing or stacking the second half** of the list.

---

## Stack-Based Key Insight

### Why Stack Works

* A stack provides **Last-In-First-Out (LIFO)** behavior.
* This perfectly matches the requirement of pulling nodes from the **end of the list**.
* If we push the **second half** of the list into a stack:

  * The top of the stack is always the next node from the back (`Ln`, `Ln-1`, ...)

### High-Level Strategy (Stack Approach)

1. **Find the middle** of the linked list using slow/fast pointers
2. **Push the second half** of the list into a stack
3. **Merge**:

   * Take one node from the front
   * Take one node from the stack
   * Rewire pointers alternately

---

## Algorithm Steps (Stack Approach)

1. **Edge case**: If list has 0 or 1 node, return
2. Use **slow and fast pointers** to find the middle
3. Push nodes from `slow.next` to end into a stack
4. Set `slow.next = None` (split the list)
5. Iterate from the head:

   * Pop from stack
   * Insert popped node after current node
6. Stop when stack is empty

---

## Python 3 Solution (With Typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        # Step 1: Find middle
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # Step 2: Push second half into stack
        stack = []
        curr = slow.next
        slow.next = None  # split the list

        while curr:
            stack.append(curr)
            curr = curr.next

        # Step 3: Merge two halves
        curr = head
        while stack:
            tail = stack.pop()
            tail.next = curr.next
            curr.next = tail
            curr = tail.next
```

---

## Worked-Out Example

### Input

```
1 → 2 → 3 → 4 → 5
```

---

### Step 1: Find Middle

```
slow = 3
```

---

### Step 2: Push Second Half into Stack

```
Stack (top → bottom): [5, 4]
Remaining list: 1 → 2 → 3
```

---

### Step 3: Merge

#### Iteration 1

```
Current: 1
Pop: 5

1 → 5 → 2 → 3
```

#### Iteration 2

```
Current: 2
Pop: 4

1 → 5 → 2 → 4 → 3
```

---

### Final Output

```
1 → 5 → 2 → 4 → 3
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(n)** (stack) |

---

## Interview Notes

* Stack solution is **easy to reason about**
* Follow-up discussion often leads to the **O(1) space solution** (reverse second half + merge)
* Demonstrates understanding of:

  * Two-pointer technique
  * Stack usage
  * In-place pointer manipulation

---

If you want, I can also explain:

* **O(1) space optimal solution**
* **Dry-run with even-length list**
* **Why stack is easier but not optimal**
* **Comparison: stack vs reverse-half approach**

Tell me how deep you want to go.