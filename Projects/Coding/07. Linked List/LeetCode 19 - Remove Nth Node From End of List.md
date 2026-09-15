---
date: "2025-12-19"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 19: Remove Nth Node From End of List"
tags:
  - leetcode
  - coding
  - linked-list
---

# LeetCode 19: Remove Nth Node From End of List

Below is a **complete, structured explanation of LeetCode 19** aligned with your usual learning format.

---

## LeetCode 19 — Remove Nth Node From End of List

### Problem Statement

You are given the head of a singly linked list and an integer `n`.  
Remove the **n-th node from the end** of the list and return the head of the modified list.

**Constraints**

* The number of nodes in the list is at least `1`.
* `1 ≤ n ≤ length of the list`.

---

## Key Observation

Removing the *n-th node from the end* is equivalent to removing the  
**(length − n + 1)-th node from the start**.

However:

* Computing the length first requires **two passes**
* We can do it in **one pass** using a clever pointer offset

---

## Core Linked List Insight (Two-Pointer Technique)

### Idea

Maintain a fixed gap of `n` nodes between two pointers:

* `fast`
* `slow`

### Strategy

1. Move `fast` pointer **n steps ahead**
2. Move both `fast` and `slow` together until `fast` reaches the end
3. At this point:

   * `slow` is just **before** the node to remove
4. Adjust pointers to skip the target node

### Why This Works

Because when `fast` reaches the end:

* `slow` has traveled `(length − n)` steps
* So `slow.next` is exactly the **n-th node from the end**

---

## Important Edge Case

When `n == length of list`, the **head must be removed**.

To handle this cleanly:

* Use a **dummy node** before the head

---

## Python 3 Implementation (with typing)

```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = dummy
        slow = dummy

        # Step 1: Move fast n steps ahead
        for _ in range(n):
            fast = fast.next

        # Step 2: Move both pointers until fast reaches the end
        while fast.next:
            fast = fast.next
            slow = slow.next

        # Step 3: Remove the nth node from end
        slow.next = slow.next.next

        return dummy.next
```

---

## Worked-Out Example

### Input

```
Linked List: 1 → 2 → 3 → 4 → 5
n = 2
```

### Step-by-Step Pointer Movement

**Initial Setup**

```
dummy → 1 → 2 → 3 → 4 → 5
fast = slow = dummy
```

**Move `fast` 2 steps**

```
fast → 2
slow → dummy
```

**Move both until `fast.next` is None**

| Step | fast | slow |
| --- | --- | --- |
| 1 | 3 | 1 |
| 2 | 4 | 2 |
| 3 | 5 | 3 |

Now:

```
slow → 3
slow.next → 4  (node to remove)
```

**Removal**

```
3.next = 5
```

### Final List

```
1 → 2 → 3 → 5
```

---

## Time & Space Complexity

* **Time Complexity:** `O(N)` (single traversal)
* **Space Complexity:** `O(1)` (constant extra space)

---

## Key Takeaway

This problem is a **canonical example** of:

* Two-pointer offset technique
* Dummy node usage to simplify head-edge cases
* One-pass linked list manipulation

If you want, I can also:

* Draw a pointer-movement table visually
* Explain why this fails without a dummy node
* Compare with the two-pass solution for intuition contrast