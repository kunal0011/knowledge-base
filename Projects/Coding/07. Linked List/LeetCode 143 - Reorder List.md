---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 143: Reorder List"
tags:
  - leetcode
  - coding
  - linked-list
  - two-pointers
  - fast-slow-pointers
  - amazon
  - google
---

# LeetCode 143: Reorder List

**Target Companies:** Amazon (Top Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Midpoint Split + In-Place Suffix Reversal + Alternating Interleaving  

---

### Problem Statement

You are given the head of a singly linked-list. The list can be represented as:
$$L_0 \to L_1 \to \dots \to L_{n - 1} \to L_n$$

Reorder the list to be on the following form:
$$L_0 \to L_n \to L_1 \to L_{n - 1} \to L_2 \to L_{n - 2} \to \dots$$

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`
- **Output:** None (modify linked list in-place)
- **Constraints:**
  - The number of nodes in the list is in the range $[1, 5 \times 10^4]$.
  - $1 \le \text{Node.val} \le 1000$

---

### Key Idea & Intuition

Rather than copying nodes into an array (which consumes $O(N)$ extra memory), this problem decomposes into three classic $O(1)$-space subroutines:

1. **Find the Middle Node (Tortoise & Hare):**
   - Advance `slow` by 1 step and `fast` by 2 steps until `fast.next` and `fast.next.next` are null.
   - `slow` lands on the end of the first half (or the exact middle for odd lists).
2. **Reverse the Second Half:**
   - Sever the first half from the second half: `second = slow.next`, `slow.next = None`.
   - Reverse the second half in-place using standard three-pointer reversal (`prev`, `curr`, `next`).
3. **Interleave the Two Lists:**
   - Alternate nodes from the first half (`first`) and the reversed second half (`second`):
     - `tmp1 = first.next`, `tmp2 = second.next`
     - `first.next = second`
     - `second.next = tmp1`
     - `first = tmp1`, `second = tmp2`

---

### Solution Approach (Step-by-Step)

1. If `head` is null or `head.next` is null, return immediately.
2. **Stage 1 (Find Midpoint):**
   - `slow = head`, `fast = head.next`.
   - While `fast` and `fast.next`:
     - `slow = slow.next`
     - `fast = fast.next.next`
3. **Stage 2 (Reverse Second Half):**
   - `second = slow.next`
   - `slow.next = None` (terminate first half)
   - `prev = None`
   - While `second`:
     - `nxt = second.next`
     - `second.next = prev`
     - `prev = second`
     - `second = nxt`
   - (Now `prev` is the head of the reversed second half).
4. **Stage 3 (Interleave):**
   - `first = head`, `second = prev`
   - While `second`:
     - `t1 = first.next`, `t2 = second.next`
     - `first.next = second`
     - `second.next = t1`
     - `first = t1`, `second = t2`

---

### Visual Algorithm Walkthrough

```
Input: [1 -> 2 -> 3 -> 4 -> 5]

Stage 1: Find Midpoint with slow & fast
  1 -> 2 -> 3 -> 4 -> 5
       slow      fast
  Split at slow (node 3):
  First half:  [1 -> 2 -> 3 -> null]
  Second half: [4 -> 5 -> null]

Stage 2: Reverse Second Half
  [4 -> 5 -> null] becomes [5 -> 4 -> null]
  Head of reversed second half: prev = [5]

Stage 3: Alternating Interleave
  First:  1 ------> 2 ------> 3 -> null
          \        / \       /
           \      /   \     /
  Second:   5 ---'     4 --'

  Step 1: 1.next = 5, 5.next = 2
  Step 2: 2.next = 4, 4.next = 3
  Second becomes null!

Final Reordered List:
  1 -> 5 -> 2 -> 4 -> 3 -> null
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Odd Length List
- **Input:** `head = [1, 2, 3, 4, 5]`
- **First Half:** `[1, 2, 3]`
- **Reversed Second:** `[5, 4]`
- **Interleaved:** `[1, 5, 2, 4, 3]`

#### Example 2: Even Length List
- **Input:** `head = [1, 2, 3, 4]`
- **First Half:** `[1, 2]`
- **Reversed Second:** `[4, 3]`
- **Interleaved:** `[1, 4, 2, 3]`

#### Example 3: Minimal 2-Node List
- **Input:** `head = [1, 2]`
- **Output:** `[1, 2]`

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
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return
            
        # 1. Find midpoint of list
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        # 2. Reverse second half
        second = slow.next
        slow.next = None  # break list into two halves
        prev = None
        while second:
            nxt = second.next
            second.next = prev
            prev = second
            second = nxt
            
        # 3. Interleave first half and reversed second half
        first = head
        second = prev
        while second:
            t1 = first.next
            t2 = second.next
            first.next = second
            second.next = t1
            first = t1
            second = t2
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
    void reorderList(ListNode* head) {
        if (!head || !head->next) return;

        // 1. Find midpoint
        ListNode* slow = head;
        ListNode* fast = head->next;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // 2. Reverse second half
        ListNode* second = slow->next;
        slow->next = nullptr;
        ListNode* prev = nullptr;
        while (second != nullptr) {
            ListNode* nxt = second->next;
            second->next = prev;
            prev = second;
            second = nxt;
        }

        // 3. Interleave
        ListNode* first = head;
        second = prev;
        while (second != nullptr) {
            ListNode* t1 = first->next;
            ListNode* t2 = second->next;
            first->next = second;
            second->next = t1;
            first = t1;
            second = t2;
        }
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
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) return;

        // 1. Find middle of list
        ListNode slow = head;
        ListNode fast = head.next;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // 2. Reverse second half
        ListNode second = slow.next;
        slow.next = null;
        ListNode prev = null;
        while (second != null) {
            ListNode nxt = second.next;
            second.next = prev;
            prev = second;
            second = nxt;
        }

        // 3. Interleave two halves
        ListNode first = head;
        second = prev;
        while (second != null) {
            ListNode t1 = first.next;
            ListNode t2 = second.next;
            first.next = second;
            second.next = t1;
            first = t1;
            second = t2;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Stage 1 traverses $N/2$ nodes; Stage 2 reverses $N/2$ nodes; Stage 3 merges $N$ nodes. Total operations are bounded by $2N = O(N)$.
- **Space Complexity:** $O(1)$ strictly in-place — No additional nodes or arrays created.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Midpoint Decomposition + Suffix Inversion + Zipper Interleave.
- **Trap:** Forgetting to sever `slow.next = None`: if you fail to set `slow.next = None`, the first half retains a cycle pointing into the second half, causing an infinite loop during interleaving!