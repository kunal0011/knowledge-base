---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 2: Add Two Numbers"
tags:
  - leetcode
  - coding
  - linked-list
  - math
  - simulation
  - amazon
  - google
---

# LeetCode 2: Add Two Numbers

**Target Companies:** Amazon (All-Time Most Asked #1), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Linked List Parallel Traversal with Carry Propagation & Sentinel Head  

---

### Problem Statement

You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

---

### Input & Output Formats & Constraints

- **Input:** `l1: Optional[ListNode]`, `l2: Optional[ListNode]`
- **Output:** `Optional[ListNode]` (head of new linked list)
- **Constraints:**
  - The number of nodes in each linked list is in the range $[1, 100]$.
  - $0 \le \text{Node.val} \le 9$
  - It is guaranteed that the list represents a number that does not have leading zeros (except $0$ itself).

---

### Key Idea & Intuition

Since digits are stored in reverse order:
- The head of each list represents the **least significant digit** (ones place).
- This mirrors standard manual grade-school column addition: we add corresponding digits column-by-column moving left-to-right, propagating a `carry` forward to the next place value.
- **Sentinel (Dummy) Node:** Creating a `dummy` head eliminates all edge cases regarding initializing the head pointer of the returned linked list.
- **Termination Invariant:** The addition loop must continue while `l1` has nodes, OR `l2` has nodes, OR `carry > 0`. This cleanly handles lists of differing lengths and creates an extra final node if the highest-order digits generate a trailing carry (e.g., $99 + 1 = 100$).

---

### Solution Approach (Step-by-Step)

1. Initialize `dummy = ListNode(0)`, `curr = dummy`, and `carry = 0`.
2. While `l1` is not null, or `l2` is not null, or `carry > 0`:
   - Extract `val1 = l1.val if l1 else 0`.
   - Extract `val2 = l2.val if l2 else 0`.
   - Compute `total = val1 + val2 + carry`.
   - Update `carry = total // 10`.
   - Create a new node with `total % 10` and attach to `curr.next`.
   - Advance `curr = curr.next`.
   - If `l1` exists, `l1 = l1.next`.
   - If `l2` exists, `l2 = l2.next`.
3. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

```
l1 = [2 -> 4 -> 3]  (342)
l2 = [5 -> 6 -> 4]  (465)

dummy -> [0]
curr = dummy, carry = 0

Step 1:
  val1 = 2, val2 = 5, carry = 0
  total = 2 + 5 + 0 = 7
  carry = 7 // 10 = 0
  digit = 7 % 10 = 7
  dummy -> [7]
  l1 -> 4, l2 -> 6

Step 2:
  val1 = 4, val2 = 6, carry = 0
  total = 4 + 6 + 0 = 10
  carry = 10 // 10 = 1
  digit = 10 % 10 = 0
  dummy -> [7 -> 0]
  l1 -> 3, l2 -> 4

Step 3:
  val1 = 3, val2 = 4, carry = 1
  total = 3 + 4 + 1 = 8
  carry = 8 // 10 = 0
  digit = 8 % 10 = 8
  dummy -> [7 -> 0 -> 8]
  l1 = null, l2 = null

Loop terminates (l1 is null, l2 is null, carry is 0).
Output list: [7 -> 0 -> 8] (represents 807 = 342 + 465).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Symmetric Lists
- **Input:** `l1 = [2, 4, 3]`, `l2 = [5, 6, 4]`
- **Output:** `[7, 0, 8]`

#### Example 2: Unequal Lengths with Trailing Carry
- **Input:** `l1 = [9, 9, 9, 9]`, `l2 = [9, 9]`
- **Trace:**
  - $9 + 9 = 18 \implies 8$, carry 1
  - $9 + 9 + 1 = 19 \implies 9$, carry 1
  - $9 + 0 + 1 = 10 \implies 0$, carry 1
  - $9 + 0 + 1 = 10 \implies 0$, carry 1
  - $0 + 0 + 1 = 1 \implies 1$, carry 0
- **Output:** `[8, 9, 0, 0, 1]` ($9999 + 99 = 10098$)

#### Example 3: Zero Lists
- **Input:** `l1 = [0]`, `l2 = [0]`
- **Output:** `[0]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            
            curr.next = ListNode(total % 10)
            curr = curr.next
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
                
        return dummy.next
```

#### 2. C++ (C++17 / STL)
```cpp
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* curr = &dummy;
        int carry = 0;

        while (l1 != nullptr || l2 != nullptr || carry != 0) {
            int val1 = (l1 != nullptr) ? l1->val : 0;
            int val2 = (l2 != nullptr) ? l2->val : 0;

            int total = val1 + val2 + carry;
            carry = total / 10;

            curr->next = new ListNode(total % 10);
            curr = curr->next;

            if (l1 != nullptr) l1 = l1->next;
            if (l2 != nullptr) l2 = l2->next;
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
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;
        int carry = 0;

        while (l1 != null || l2 != null || carry != 0) {
            int val1 = (l1 != null) ? l1.val : 0;
            int val2 = (l2 != null) ? l2.val : 0;

            int total = val1 + val2 + carry;
            carry = total / 10;

            curr.next = new ListNode(total % 10);
            curr = curr.next;

            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }

        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\max(M, N))$ where $M$ and $N$ are the lengths of `l1` and `l2`. The loop runs at most $\max(M, N) + 1$ times.
- **Space Complexity:** $O(\max(M, N))$ to allocate the new linked list representing the resulting sum.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Sentinel/Dummy Head Pattern + Parallel Linked List Traversal.
- **Trap:** Forgetting the trailing `carry`! If `l1` and `l2` are exhausted but `carry == 1` (e.g. $5 + 5 = 10$), omitting `carry` from the `while` condition causes the most significant digit `1` to be dropped.