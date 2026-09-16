---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 206: Reverse Linked List"
tags:
  - leetcode
  - coding
  - linked-list
  - recursion
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 206: Reverse Linked List

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** In-Place Pointer Reversal / Iterative & Recursive

---

### Problem Statement

Given the `head` of a singly linked list, reverse the list, and return *the reversed list*.

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`
- **Output:** `Optional[ListNode]` (new head of the reversed list)
- **Constraints:**
  - The number of nodes in the list is the range $[0, 5000]$.
  - $-5000 \le \text{Node.val} \le 5000$
- **Follow up:** A linked list can be reversed either iteratively or recursively. Could you implement both?

---

### Key Idea & Intuition

#### 1. The Three-Pointer Invariant:
In a singly linked list, each node stores a pointer only to its successor (`curr.next`). Reversing the link direction (`curr.next = prev`) destroys the reference to the subsequent node!
Therefore, to reverse each pointer safely in-place without losing the remainder of the list, we need **three pointers**:
1. `prev`: The node that will become `curr`'s new next node (initially `None`).
2. `curr`: The node whose pointer is currently being inverted (initially `head`).
3. `nxt`: A temporary placeholder preserving `curr.next` before we overwrite it.

#### 2. Reversal Loop Invariant:
At each iteration:
```
nxt = curr.next    # 1. Save future
curr.next = prev   # 2. Reverse arrow
prev = curr        # 3. Advance prev
curr = nxt         # 4. Advance curr
```
When `curr` reaches `None`, `prev` points to the last non-null node, which is the new `head` of the reversed list.

#### 3. Recursive Formulation:
If we recursively reverse `head.next`:
- The recursive call returns the new head `new_head`.
- `head.next` is now the tail of that reversed sublist.
- To attach `head` after it: `head.next.next = head`.
- Disconnect old forward pointer: `head.next = None`.
- Return `new_head`.

---

### Solution Approach (Step-by-Step)

#### Iterative Approach (Recommended for $\mathcal{O}(1)$ Space):
1. **Initialize Pointers:**
   - `prev = None`, `curr = head`.
2. **Iterate While `curr is not None`:**
   - Save next node: `nxt = curr.next`.
   - Reverse link: `curr.next = prev`.
   - Shift `prev` forward: `prev = curr`.
   - Shift `curr` forward: `curr = nxt`.
3. **Return New Head:**
   - Return `prev`.

---

### Visual Algorithm Walkthrough

#### Example: `1 -> 2 -> 3 -> None`

```
Initial:
prev = None, curr = 1

Step 1:
  nxt = 2
  1.next = None (reversed!)
  prev = 1
  curr = 2
  State: None <- 1    2 -> 3 -> None

Step 2:
  nxt = 3
  2.next = 1 (reversed!)
  prev = 2
  curr = 3
  State: None <- 1 <- 2    3 -> None

Step 3:
  nxt = None
  3.next = 2 (reversed!)
  prev = 3
  curr = None
  State: None <- 1 <- 2 <- 3

curr is None -> Terminate.
Return prev (Node 3): 3 -> 2 -> 1 -> None.
```

---

### Solved Examples with Multiple Inputs

| Input `head` | Reversal Process | Output List | Notes |
| :--- | :--- | :--- | :--- |
| `[1, 2, 3, 4, 5]` | 5 step pointer inversions | `[5, 4, 3, 2, 1]` | Standard multi-node list |
| `[1, 2]` | 2 step pointer inversions | `[2, 1]` | Two nodes |
| `[]` | `curr = None` immediately | `[]` | Empty list base case |
| `[1]` | 1 step pointer inversion | `[1]` | Single node list |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
            
        return prev

    def reverseListRecursive(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        
        new_head = self.reverseListRecursive(head.next)
        head.next.next = head
        head.next = None
        return new_head
```

#### 2. C++ (C++17 / STL)
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */

class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* curr = head;

        while (curr != nullptr) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }

        return prev;
    }

    ListNode* reverseListRecursive(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }

        ListNode* newHead = reverseListRecursive(head->next);
        head->next->next = head;
        head->next = nullptr;
        return newHead;
    }
};
```

#### 3. Java (Modern, Typed)
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */

class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;

        while (curr != null) {
            ListNode nxt = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nxt;
        }

        return prev;
    }

    public ListNode reverseListRecursive(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode newHead = reverseListRecursive(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```

---

### Complexity Analysis

- **Iterative Approach:**
  - **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the number of nodes in the linked list. We traverse each node exactly once.
  - **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only three pointers are used.
- **Recursive Approach:**
  - **Time Complexity:** $\mathcal{O}(n)$.
  - **Space Complexity:** $\mathcal{O}(n)$ call stack memory due to recursion depth $n$.

---

### Takeaway Pattern & Interview Traps

1. **Overwriting `curr.next` Before Saving:**
   - Performing `curr.next = prev` before `nxt = curr.next` severs the reference to the rest of the list, permanently leaking nodes. Always store `nxt = curr.next` first.
2. **Cycle Creation in Recursive Reversal:**
   - In recursive reversal, forgetting `head.next = None` leaves a cycle between `head` and `head.next`, leading to an infinite loop when traversing the reversed list.
