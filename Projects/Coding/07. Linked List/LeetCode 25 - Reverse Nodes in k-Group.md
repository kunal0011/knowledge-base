---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 25: Reverse Nodes in k-Group"
tags:
  - leetcode
  - coding
  - linked-list
  - recursion
  - two-pointers
  - amazon
  - google
---

# LeetCode 25: Reverse Nodes in k-Group

**Target Companies:** Amazon (Signature Hard Classic), Google, Microsoft, Meta, Apple  
**Difficulty:** Hard  
**Topic:** Segmented In-Place Pointer Inversion with Group Anchors  

---

### Problem Statement

Given the `head` of a linked list, reverse the nodes of the list $k$ at a time, and return the modified list.

$k$ is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of $k$ then left-out nodes, in the end, should remain as it is.

You may not alter the values in the list's nodes, only nodes themselves may be changed.

---

### Input & Output Formats & Constraints

- **Input:** `head: Optional[ListNode]`, `k: int`
- **Output:** `Optional[ListNode]` (head of modified linked list)
- **Constraints:**
  - The number of nodes in the list is $n$.
  - $1 \le k \le n \le 5000$
  - $0 \le \text{Node.val} \le 1000$

---

### Key Idea & Intuition

Reversing nodes $k$ at a time requires carefully managing the boundaries between successive segments.
For every group of $k$ nodes:
1. **Lookahead Check:** Find the $k$-th node ahead from `group_prev`. If fewer than $k$ nodes remain before reaching `null`, we must leave this trailing segment unchanged and terminate.
2. **Boundary Anchors:**
   - `group_prev`: The node immediately preceding the current group.
   - `kth`: The last node of the current group (which will become the new head of the reversed group).
   - `group_next = kth.next`: The first node of the following group.
3. **In-Place Group Reversal:**
   Reverse the sublist starting from `group_prev.next` up to `kth`.
   Set `prev = group_next` as the starting tail so that the reversed sublist automatically links to the next group!
4. **Reconnect & Advance:**
   Save the old group head (`group_prev.next`, which is now the group tail).
   Point `group_prev.next = kth`.
   Advance `group_prev` to the saved old head.

---

### Solution Approach (Step-by-Step)

1. Create a sentinel node: `dummy = ListNode(0, head)`.
2. Set `group_prev = dummy`.
3. Helper `get_kth(curr, k)`: advances `curr` by $k$ steps; returns node or `None` if end is reached early.
4. While `True`:
   - `kth = get_kth(group_prev, k)`.
   - If `kth` is `None`: break loop (fewer than $k$ nodes remain).
   - `group_next = kth.next`.
   - Reverse the $k$ nodes:
     - `prev = group_next`, `curr = group_prev.next`.
     - While `curr != group_next`:
       - `tmp = curr.next`
       - `curr.next = prev`
       - `prev = curr`
       - `curr = tmp`
   - Reconnect:
     - `old_head = group_prev.next`
     - `group_prev.next = kth`
     - `group_prev = old_head`
5. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

```
List: [1 -> 2 -> 3 -> 4 -> 5], k = 2
dummy -> [1 -> 2 -> 3 -> 4 -> 5]
group_prev = dummy

Round 1 (Group [1, 2]):
1. kth node from group_prev is [2].
   group_next = kth.next = [3].
2. Reverse [1, 2] pointing into group_next [3]:
   curr = [1], prev = [3]
   - [1].next = [3], prev = [1], curr = [2]
   - [2].next = [1], prev = [2], curr = [3] (reached group_next!)
3. Reconnect:
   group_prev.next = kth ([2])
   group_prev moves to [1]
   Result after Round 1:
   dummy -> [2 -> 1] -> [3 -> 4 -> 5]
                  ^
              group_prev

Round 2 (Group [3, 4]):
1. kth node from group_prev is [4].
   group_next = [5].
2. Reverse [3, 4] pointing into [5]:
   - [4 -> 3 -> 5]
3. Reconnect:
   [1].next = [4]
   group_prev moves to [3]
   Result after Round 2:
   dummy -> [2 -> 1 -> 4 -> 3] -> [5]

Round 3:
1. kth node from [3] with k = 2:
   Only [5] exists (1 node < 2). get_kth returns null.
2. Break! Leftover [5] stays untouched.

Final: [2 -> 1 -> 4 -> 3 -> 5]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Grouping ($k = 2$)
- **Input:** `head = [1, 2, 3, 4, 5]`, `k = 2`
- **Output:** `[2, 1, 4, 3, 5]`

#### Example 2: Grouping with Larger Remainder ($k = 3$)
- **Input:** `head = [1, 2, 3, 4, 5]`, `k = 3`
- **Trace:**
  - First 3 nodes `[1, 2, 3]` reversed $\to$ `[3, 2, 1]`.
  - Remaining nodes `[4, 5]` length is 2 $< 3$, left as-is.
- **Output:** `[3, 2, 1, 4, 5]`

#### Example 3: $k = 1$ (Identity)
- **Input:** `head = [1, 2]`, `k = 1`
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
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy
        
        while True:
            kth = self._get_kth(group_prev, k)
            if not kth:
                break
                
            group_next = kth.next
            
            # Reverse group of k nodes
            prev = group_next
            curr = group_prev.next
            while curr != group_next:
                nxt = curr.next
                curr.next = prev
                prev = curr
                curr = nxt
                
            # Reconnect group_prev to new head of group
            old_head = group_prev.next
            group_prev.next = kth
            group_prev = old_head
            
        return dummy.next

    def _get_kth(self, curr: Optional[ListNode], k: int) -> Optional[ListNode]:
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr
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
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0, head);
        ListNode* groupPrev = &dummy;

        while (true) {
            ListNode* kth = getKth(groupPrev, k);
            if (!kth) break;

            ListNode* groupNext = kth->next;

            // In-place reverse
            ListNode* prev = groupNext;
            ListNode* curr = groupPrev->next;
            while (curr != groupNext) {
                ListNode* nxt = curr->next;
                curr->next = prev;
                prev = curr;
                curr = nxt;
            }

            ListNode* oldHead = groupPrev->next;
            groupPrev->next = kth;
            groupPrev = oldHead;
        }

        return dummy.next;
    }

private:
    ListNode* getKth(ListNode* curr, int k) {
        while (curr != nullptr && k > 0) {
            curr = curr->next;
            k--;
        }
        return curr;
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
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0, head);
        ListNode groupPrev = dummy;

        while (true) {
            ListNode kth = getKth(groupPrev, k);
            if (kth == null) break;

            ListNode groupNext = kth.next;

            // In-place group reversal
            ListNode prev = groupNext;
            ListNode curr = groupPrev.next;
            while (curr != groupNext) {
                ListNode nxt = curr.next;
                curr.next = prev;
                prev = curr;
                curr = nxt;
            }

            ListNode oldHead = groupPrev.next;
            groupPrev.next = kth;
            groupPrev = oldHead;
        }

        return dummy.next;
    }

    private ListNode getKth(ListNode curr, int k) {
        while (curr != null && k > 0) {
            curr = curr.next;
            k--;
        }
        return curr;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each node is visited twice: once during the $k$-th lookahead scan and once during in-place reversal. Overall runtime is strictly linear.
- **Space Complexity:** $O(1)$ auxiliary space — Strict in-place pointer modifications with no dynamic allocations or recursion stack frames.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Segmented In-Place Pointer Inversion with Group Anchors (`groupPrev`, `kth`, `groupNext`).
- **Trap:** Forgetting that leftover nodes ($< k$) must remain **unaltered**. If you reverse before counting $k$ nodes, you have to reverse the tail a second time to restore its original order. Looking ahead with `get_kth()` guarantees leftover nodes are never touched.