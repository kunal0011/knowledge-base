---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 19: Remove Nth Node From End of List"
tags:
  - leetcode
  - coding
  - linked-list
  - two-pointers
  - dummy-node
  - amazon
  - google
---

# LeetCode 19: Remove Nth Node From End of List

**Target Companies:** Amazon, Google, Meta, Apple, Microsoft  
**Difficulty:** Medium  
**Topic:** Fixed-Gap Two Pointers (Lead & Follow) with Sentinel Head  

---

### Problem Statement

Given the `head` of a linked list, remove the $n$-th node from the end of the list and return its head.

Could you do this in **one pass**?

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`, `n: int`
- **Output:** `Optional[ListNode]` (head of modified linked list)
- **Constraints:**
  - The number of nodes in the list is $sz$.
  - $1 \le sz \le 30$
  - $0 \le \text{Node.val} \le 100$
  - $1 \le n \le sz$

---

### Key Idea & Intuition

A naive solution counts the total length $L$ in a first pass, then traverses $L - n$ nodes in a second pass to delete the target node.

To accomplish this in a **single pass**:
1. We introduce a **fixed gap of $n$ steps** between a `fast` and a `slow` pointer.
2. Advance `fast` by $n$ steps forward.
3. Advance both `fast` and `slow` one step at a time until `fast` reaches the last node of the list (`fast.next == null`).
4. Because the gap between `slow` and `fast` is exactly $n$, when `fast` is at the $L$-th node (the end), `slow` will be at index $L - n$, which is **immediately preceding** the node to be removed!
5. We delete the target node simply by relinking: `slow.next = slow.next.next`.
6. **Sentinel (Dummy) Node:** Attaching a `dummy` node before `head` elegantly handles the edge case where the node to be removed is the `head` itself ($n = sz$).

---

### Solution Approach (Step-by-Step)

1. Create a sentinel node: `dummy = ListNode(0, head)`.
2. Initialize `fast = dummy` and `slow = dummy`.
3. Advance `fast` forward by $n$ steps.
4. While `fast.next` is not null:
   - Advance `fast = fast.next`.
   - Advance `slow = slow.next`.
5. Remove the $n$-th node from the end:
   - `slow.next = slow.next.next`.
6. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

```
List: [1 -> 2 -> 3 -> 4 -> 5], n = 2

Step 0: Setup dummy
dummy -> [1] -> [2] -> [3] -> [4] -> [5] -> null
slow, fast at dummy

Step 1: Move fast n = 2 steps forward
dummy -> [1] -> [2] -> [3] -> [4] -> [5] -> null
  ^               ^
 slow            fast

Step 2: Advance both until fast.next is null
- Move 1: slow at [1], fast at [3]
- Move 2: slow at [2], fast at [4]
- Move 3: slow at [3], fast at [5] (fast.next is null! STOP)

dummy -> [1] -> [2] -> [3] -> [4] -> [5] -> null
                        ^               ^
                       slow            fast

Step 3: Relink slow.next to skip [4]
slow.next = slow.next.next ([3].next points directly to [5])
Node [4] is unlinked and deleted!

Result: dummy.next -> [1 -> 2 -> 3 -> 5]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Intermediate Node Removal
- **Input:** `head = [1, 2, 3, 4, 5]`, `n = 2`
- **Output:** `[1, 2, 3, 5]`

#### Example 2: Single Node List (Removing Only Element)
- **Input:** `head = [1]`, `n = 1`
- **Trace:**
  - `fast` moves 1 step to `[1]`.
  - `fast.next` is null $\implies$ loop does not run.
  - `slow` is at `dummy`. `slow.next = dummy.next.next = null`.
- **Output:** `[]`

#### Example 3: Removing the Head Node ($n = \text{length}$)
- **Input:** `head = [1, 2]`, `n = 2`
- **Trace:**
  - `fast` moves 2 steps to `[2]`.
  - `fast.next` is null $\implies$ loop stops immediately.
  - `slow` is at `dummy`. `dummy.next = dummy.next.next = [2]`.
- **Output:** `[2]`

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
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = dummy
        slow = dummy
        
        # 1. Advance fast pointer n steps
        for _ in range(n):
            fast = fast.next
            
        # 2. Advance both until fast reaches the last node
        while fast.next:
            fast = fast.next
            slow = slow.next
            
        # 3. Delete the target node
        slow.next = slow.next.next
        
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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);
        ListNode* fast = &dummy;
        ListNode* slow = &dummy;

        for (int i = 0; i < n; ++i) {
            fast = fast->next;
        }

        while (fast->next != nullptr) {
            fast = fast->next;
            slow = slow->next;
        }

        ListNode* nodeToDelete = slow->next;
        slow->next = slow->next->next;
        delete nodeToDelete; // Prevent memory leak

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
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode fast = dummy;
        ListNode slow = dummy;

        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(L)$ where $L$ is the number of nodes in the list. Traverses the list in a single pass.
- **Space Complexity:** $O(1)$ auxiliary space — Only two pointer references are maintained.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Offset Lead-and-Follow Two Pointers with Dummy Node.
- **Trap:** Halting condition: Make sure `while (fast.next != null)` stops when `fast` is on the **last node**, so that `slow` stops on the node **before** the target node. If you wrote `while (fast != null)`, `slow` would land directly on the target node, making deletion awkward without a `prev` reference.