---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 92: Reverse Linked List II"
tags:
  - leetcode
  - coding
  - linked-list
  - amazon
  - google
---

# LeetCode 92: Reverse Linked List II

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** 1-Pass In-Place Linked List Reversal

---

### Problem Statement

Given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return the reversed list.

Follow up: Could you do it in **one pass**?

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`, `left: int`, `right: int` (1-indexed)
- **Output:** `Optional[ListNode]`
- **Constraints:**
  - The number of nodes in the list is $n$.
  - $1 \le n \le 500$
  - $-500 \le \text{Node.val} \le 500$
  - $1 \le \text{left} \le \text{right} \le n$

---

### Key Idea & Intuition

- **Sentinel Dummy Node:**
  - Reversal might start at the very head (`left = 1`). A dummy node `dummy.next = head` unifies edge cases.
- **Single-Pass In-Place Insertion (Sublist Head Insertion):**
  - Walk `prev` to node `left - 1`.
  - Let `curr = prev.next` (this node will become the tail of the reversed subsegment).
  - For $right - left$ iterations:
    - Take the node immediately after `curr` (`then = curr.next`).
    - Splice `then` out of its current position.
    - Move `then` directly after `prev`.
    - This gradually moves each subsequent node to the front of the sublist!

---

### Solution Approach (Step-by-Step)

1. Create `dummy = ListNode(0, head)`. Set `prev = dummy`.
2. Move `prev` forward $left - 1$ times so it sits right before the reversal window.
3. Let `curr = prev.next`.
4. For $i$ in range $right - left$:
   - `then = curr.next`
   - `curr.next = then.next`
   - `then.next = prev.next`
   - `prev.next = then`
5. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

```
List: 1 -> 2 -> 3 -> 4 -> 5, left = 2, right = 4
prev = 1, curr = 2

Iteration 1 (move 3 after prev):
then = 3
curr.next = 4
then.next = 2
prev.next = 3
State: 1 -> 3 -> 2 -> 4 -> 5

Iteration 2 (move 4 after prev):
then = 4
curr.next = 5
then.next = 3
prev.next = 4
State: 1 -> 4 -> 3 -> 2 -> 5

Window reversed in 1 pass!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Reversal in Middle
- **Input:** `head = [1,2,3,4,5], left = 2, right = 4`
- **Output:** `[1,4,3,2,5]`

#### Example 2: Reversal from Head (`left = 1`)
- **Input:** `head = [5], left = 1, right = 1`
- **Output:** `[5]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if not head or left == right:
            return head
            
        dummy = ListNode(0, head)
        prev = dummy
        
        for _ in range(left - 1):
            prev = prev.next
            
        curr = prev.next
        for _ in range(right - left):
            then = curr.next
            curr.next = then.next
            then.next = prev.next
            prev.next = then
            
        return dummy.next
```

#### 2. C++ (C++17 / STL)
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x, ListNode *n = nullptr) : val(x), next(n) {}
};

class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        if (!head || left == right) return head;

        ListNode dummy(0, head);
        ListNode* prev = &dummy;

        for (int i = 0; i < left - 1; ++i) {
            prev = prev->next;
        }

        ListNode* curr = prev->next;
        for (int i = 0; i < right - left; ++i) {
            ListNode* then = curr->next;
            curr->next = then->next;
            then->next = prev->next;
            prev->next = then;
        }

        return dummy.next;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class ListNode {
    int val;
    ListNode next;
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if (head == null || left == right) return head;

        ListNode dummy = new ListNode(0, head);
        ListNode prev = dummy;

        for (int i = 0; i < left - 1; i++) {
            prev = prev.next;
        }

        ListNode curr = prev.next;
        for (int i = 0; i < right - left; i++) {
            ListNode then = curr.next;
            curr.next = then.next;
            then.next = prev.next;
            prev.next = then;
        }

        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Exactly one single pass through the list.
- **Space Complexity:** $O(1)$ — Only pointer manipulation in-place.

---

### Takeaway Pattern & Interview Traps

1. **The Four-Pointer Splicing Invariant:**
   - In sublist reversal, moving nodes one-by-one to the head of the sublist avoids multiple passes:
     - `then = curr.next` (isolate next node to bring forward)
     - `curr.next = then.next` (bridge across `then`)
     - `then.next = prev.next` (point `then` to current sublist head)
     - `prev.next = then` (attach `then` after `prev`)
2. **Sentinel Dummy Node Necessity:**
   - When `left = 1`, the head itself changes. Having `dummy.next = head` provides a stable `prev` pointer right before the head, allowing uniform code without special-casing `left == 1`.

