---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 141: Linked List Cycle"
tags:
  - leetcode
  - coding
  - linked-list
  - two-pointers
  - floyds-tortoise-and-hare
  - cycle-detection
  - amazon
  - google
---

# LeetCode 141: Linked List Cycle

**Target Companies:** Amazon (Top Classic), Microsoft, Google, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** Floyd's Tortoise and Hare / Fast & Slow Pointers Cycle Detection  

---

### Problem Statement

Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter.**

Return `true` if there is a cycle in the linked list. Otherwise, return `false`.

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`
- **Output:** `bool` (`True` if a cycle exists, `False` otherwise)
- **Constraints:**
  - The number of the nodes in the list is in the range $[0, 10^4]$.
  - $-10^5 \le \text{Node.val} \le 10^5$
  - `pos` is `-1` or a valid index in the linked list.

---

### Key Idea & Intuition

#### Why Hash Set is Suboptimal ($O(N)$ Space):
Storing visited node addresses in a hash set takes $O(N)$ auxiliary memory.

#### Floyd's Cycle-Finding Algorithm ($O(N)$ Time, $O(1)$ Space):
We place two pointers at `head`:
- `slow`: moves **1 step** per iteration (`slow = slow.next`).
- `fast`: moves **2 steps** per iteration (`fast = fast.next.next`).

**Mathematical Proof of Convergence:**
1. If no cycle exists, `fast` or `fast.next` will encounter `null` in at most $N/2$ steps, terminating immediately.
2. If a cycle of length $C$ exists, both pointers will eventually enter the cycle. Once `slow` enters the cycle, let the distance from `slow` to `fast` moving forward along the cycle be $d \in [0, C - 1]$.
3. In each iteration, `slow` advances by 1 and `fast` advances by 2. The relative distance between them decreases by:
   $$(d + 2 - 1) \pmod C = (d + 1) \pmod C$$
   Viewed from `fast` catching up to `slow` from behind, the gap shrinks by exactly **1 node per iteration**.
4. Since the gap decreases by 1 in every single step, `fast` is strictly guaranteed to meet `slow` in at most $C$ iterations without ever jumping over it!

---

### Solution Approach (Step-by-Step)

1. Check if `head` is null or `head.next` is null: return `False`.
2. Initialize `slow = head`, `fast = head`.
3. While `fast` is not null and `fast.next` is not null:
   - Advance `slow = slow.next`.
   - Advance `fast = fast.next.next`.
   - If `slow == fast`: return `True` (cycle detected).
4. If loop terminates by encountering `null`, return `False`.

---

### Visual Algorithm Walkthrough

```
List: 3 -> 2 -> 0 -> -4
           ^          |
           |----------| (pos = 1, cycle length C = 3)

Initial: slow = 3, fast = 3

Step 1:
  slow = slow.next = 2
  fast = fast.next.next = 0
  slow (2) != fast (0)

Step 2:
  slow = slow.next = 0
  fast = fast.next.next: 0 -> -4 -> 2
  slow (0) != fast (2)

Step 3:
  slow = slow.next = -4
  fast = fast.next.next: 2 -> 0 -> -4
  slow (-4) == fast (-4)!

Collision detected at node with value -4!
Return True.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Multi-Node Cycle
- **Input:** `head = [3, 2, 0, -4]`, `pos = 1`
- **Output:** `true`

#### Example 2: Two-Node Cycle
- **Input:** `head = [1, 2]`, `pos = 0`
- **Trace:**
  - `slow = 1, fast = 1`
  - Step 1: `slow = 2, fast = 1`
  - Step 2: `slow = 1, fast = 1` $\implies slow == fast$.
- **Output:** `true`

#### Example 3: Single Node No Cycle
- **Input:** `head = [1]`, `pos = -1`
- **Trace:** `fast.next == null` $\implies$ immediately returns `false`.
- **Output:** `false`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class ListNode:
    def __init__(self, x: int):
        self.val = x
        self.next: Optional['ListNode'] = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow = head;
        ListNode *fast = head;

        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                return true;
            }
        }

        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int x) {
        val = x;
        next = null;
    }
}

public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                return true;
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — If there is no cycle, `fast` reaches the end in $N/2$ steps. If there is a cycle of length $C$, the number of iterations before `slow` enters the cycle is at most $N$, and `fast` catches up to `slow` in at most $C$ iterations ($C \le N$). Total time is bounded by $O(N)$.
- **Space Complexity:** $O(1)$ — Strictly constant extra space with two pointers.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Floyd's Tortoise and Hare Pointer Disparity.
- **Trap:** `NullPointerException` on `fast.next.next`: Always verify both `while (fast != null && fast.next != null)` before dereferencing `fast.next.next`.