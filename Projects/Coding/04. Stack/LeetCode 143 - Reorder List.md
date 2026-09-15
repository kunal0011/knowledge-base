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
  - linked-list
  - two-pointers
  - amazon
  - google
---

# LeetCode 143: Reorder List

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Stack / Linked List / In-Place Reversal

---

### Problem Statement

You are given the head of a singly linked-list. The list can be represented as:

$$L_0 \to L_1 \to \dots \to L_{n - 1} \to L_n$$

Reorder the list to be on the following form:

$$L_0 \to L_n \to L_1 \to L_{n - 1} \to L_2 \to L_{n - 2} \to \dots$$

You may not modify the values in the list's nodes. Only nodes themselves may be changed (i.e., you must rewire node pointers in-place).

---

### Input & Output Formats & Constraints

- **Input:**
  - `head`: `Optional[ListNode]`, the head of a singly-linked list.
- **Output:**
  - `None`: The list must be modified in-place.
- **Constraints:**
  - The number of nodes in the list is in the range $[1, 5 \times 10^4]$.
  - $1 \le \text{Node.val} \le 1000$.

---

### Key Idea & Intuition

The desired ordering alternates between nodes from the **front** and nodes from the **back**:
- $L_0$ (1st from front) $\to L_n$ (1st from back) $\to L_1$ (2nd from front) $\to L_{n-1}$ (2nd from back) $\dots$

Because singly-linked lists do not support backward traversal, accessing nodes in reverse order requires either:
1. **LIFO Stack ($\mathcal{O}(N)$ Space):**
   - Push nodes from the second half (or the entire list) onto a stack.
   - Popping from the stack yields the nodes from the back in reverse order ($L_n, L_{n-1}, \dots$).
   - We interleave nodes from the front of the list with popped nodes from the stack until the middle is reached.
2. **Reverse Second Half In-Place ($\mathcal{O}(1)$ Space - Optimal Interview Standard):**
   - Step 1: Find the midpoint of the linked list using slow and fast pointers.
   - Step 2: Split the list into two halves and reverse the second half in-place.
   - Step 3: Merge the first half and the reversed second half alternately.

---

### Solution Approach (Step-by-Step)

#### Approach 1: Stack Approach ($\mathcal{O}(N)$ Space)
1. Use slow/fast pointers to find the middle of the list.
2. Push all nodes in the second half (`slow.next` to the end) into a stack.
3. Sever the first half from the second half: `slow.next = None`.
4. Iterate from `curr = head`:
   - Pop `tail = stack.pop()`.
   - Save `next_front = curr.next`.
   - Wire `curr.next = tail`, `tail.next = next_front`.
   - Advance `curr = next_front`.

#### Approach 2: Three-Step Optimal In-Place ($\mathcal{O}(1)$ Space)
1. **Find Middle:**
   ```
   slow, fast = head, head
   while fast and fast.next:
       slow = slow.next
       fast = fast.next.next
   ```
2. **Reverse Second Half:**
   - Detach `second = slow.next` and set `slow.next = None`.
   - Reverse `second` using standard three-pointer reversal (`prev`, `curr`, `nxt`).
3. **Interleave Two Halves:**
   - Maintain `first = head` and `second = prev` (head of reversed second half).
   - Alternately splice nodes from `first` and `second`.

---

### Visual Algorithm Walkthrough

Suppose `head = 1 -> 2 -> 3 -> 4 -> 5`:

```
Step 1: Locate Middle
  1 -> 2 -> 3 -> 4 -> 5
            ^
           slow (middle)

Step 2: Split and Reverse Second Half
  First Half:   1 -> 2 -> 3 -> null
  Second Half:  4 -> 5 -> null
  Reversed:     5 -> 4 -> null

Step 3: Interleave (Merge)
  First:   1 -> 2 -> 3
  Second:  5 -> 4

  Iteration 1:
    1 connects to 5
    5 connects to 2
    List so far: 1 -> 5 -> 2 -> 3

  Iteration 2:
    2 connects to 4
    4 connects to 3
    List so far: 1 -> 5 -> 2 -> 4 -> 3

Final Result: 1 -> 5 -> 2 -> 4 -> 3
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Odd Length List

- **Input:** `1 -> 2 -> 3 -> 4 -> 5`
- **Tracing (Three-Step Method):**
  - Middle: `3`.
  - First half: `1 -> 2 -> 3 -> null`.
  - Second half reversed: `5 -> 4 -> null`.
  - Merge:
    - $1 \to 5 \to 2$
    - $2 \to 4 \to 3$
- **Output:** `1 -> 5 -> 2 -> 4 -> 3`

#### Example 2: Even Length List

- **Input:** `1 -> 2 -> 3 -> 4`
- **Tracing:**
  - `slow` lands at node `2` (or `3` depending on termination condition).
  - First half: `1 -> 2 -> null`.
  - Second half reversed: `4 -> 3 -> null`.
  - Merge:
    - $1 \to 4 \to 2$
    - $2 \to 3 \to \text{null}$
- **Output:** `1 -> 4 -> 2 -> 3`

#### Example 3: Edge Cases (Length $\le 2$)

- **Input:** `1 -> 2`
  - Already in form $L_0 \to L_1$. Remains `1 -> 2`.
- **Input:** `1`
  - Single node; returns unchanged.

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(1)$ Space Implementation
```python
from typing import Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Reorders list in-place using O(1) auxiliary space.
        """
        if not head or not head.next or not head.next.next:
            return

        # 1. Find the middle node
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 2. Reverse the second half
        prev = None
        curr = slow.next
        slow.next = None  # Cut the first half

        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        second = prev  # Head of reversed second half

        # 3. Interleave first half and reversed second half
        first = head
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first = tmp1
            second = tmp2
```

##### Stack-Based Implementation ($\mathcal{O}(N)$ Space)
```python
class SolutionStack:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        # Push second half into stack
        stack = []
        curr = slow.next
        slow.next = None

        while curr:
            stack.append(curr)
            curr = curr.next

        curr = head
        while stack:
            tail = stack.pop()
            tail.next = curr.next
            curr.next = tail
            curr = tail.next
```

#### C++17

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
        if (!head || !head->next || !head->next->next) return;

        // 1. Find midpoint (slow/fast pointers)
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast != nullptr && fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }

        // 2. Reverse second half
        ListNode* prev = nullptr;
        ListNode* curr = slow->next;
        slow->next = nullptr; // Disconnect first half

        while (curr != nullptr) {
            ListNode* nxt = curr->next;
            curr->next = prev;
            prev = curr;
            curr = nxt;
        }

        // 3. Interleave two halves
        ListNode* first = head;
        ListNode* second = prev;
        while (second != nullptr) {
            ListNode* tmp1 = first->next;
            ListNode* tmp2 = second->next;

            first->next = second;
            second->next = tmp1;

            first = tmp1;
            second = tmp2;
        }
    }
};
```

#### Java

```java
public class Solution {
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) return;

        // 1. Find middle of the linked list
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // 2. Reverse the second half
        ListNode prev = null;
        ListNode curr = slow.next;
        slow.next = null; // Sever first half

        while (curr != null) {
            ListNode nxt = curr.next;
            curr.next = prev;
            prev = curr;
            curr = nxt;
        }

        // 3. Interleave the two lists
        ListNode first = head;
        ListNode second = prev;
        while (second != null) {
            ListNode tmp1 = first.next;
            ListNode tmp2 = second.next;

            first.next = second;
            second.next = tmp1;

            first = tmp1;
            second = tmp2;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Finding the middle takes $\mathcal{O}(N)$ (traversing $N$ nodes).
  - Reversing the second half takes $\mathcal{O}(N/2) = \mathcal{O}(N)$ time.
  - Merging the two halves takes $\mathcal{O}(N/2) = \mathcal{O}(N)$ time.
  - Total Time: $\mathcal{O}(N)$, linear in the number of nodes.
- **Space Complexity:**
  - **Optimal Reversal Approach:** $\mathcal{O}(1)$ auxiliary space as pointers are manipulated strictly in-place.
  - **Stack Approach:** $\mathcal{O}(N)$ auxiliary space to hold the second half of nodes.

---

### Takeaway Pattern & Interview Traps

1. **The Three Core Linked List Sub-problems:**
   - LeetCode 143 is famous because it tests three fundamental linked list operations in a single question:
     1. Finding the midpoint with slow/fast pointers (LC 876).
     2. Reversing a singly linked list in-place (LC 206).
     3. Merging two linked lists alternately (similar to LC 21).
2. **Severing the List (`slow.next = null`):**
   - Failing to disconnect `slow.next` will cause a cycle in the list when merging, resulting in an infinite loop.
3. **Saving Pointer References Before Rewiring:**
   - Always cache `tmp1 = first.next` and `tmp2 = second.next` *before* mutating `first.next = second` to prevent orphaned pointers.