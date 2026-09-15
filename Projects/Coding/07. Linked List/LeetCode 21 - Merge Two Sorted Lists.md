---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 21: Merge Two Sorted Lists"
tags:
  - leetcode
  - coding
  - linked-list
  - two-pointers
  - merge-sort
  - amazon
  - google
---

# LeetCode 21: Merge Two Sorted Lists

**Target Companies:** Amazon (Top Classic), Microsoft, Google, Apple, Meta  
**Difficulty:** Easy  
**Topic:** Sentinel Node In-Place Pointer Splice  

---

### Problem Statement

You are given the heads of two sorted linked lists `list1` and `list2`.

Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.

Return the **head of the merged linked list**.

---

### Input & Output Formats & Constraints

- **Input:** `list1: Optional[ListNode]`, `list2: Optional[ListNode]`
- **Output:** `Optional[ListNode]` (head of merged sorted list)
- **Constraints:**
  - The number of nodes in both lists is in the range $[0, 50]$.
  - $-100 \le \text{Node.val} \le 100$
  - Both `list1` and `list2` are sorted in **non-decreasing** order.

---

### Key Idea & Intuition

Because both input lists are already sorted in non-decreasing order, this is the fundamental `merge` subroutine of **Merge Sort**:
1. We maintain a pointer `tail` to the last attached node of our merged list.
2. At each iteration, we compare `list1.val` and `list2.val`.
3. We splice the node with the smaller (or equal) value directly onto `tail.next`, and advance that list's pointer.
4. **Sentinel (Dummy) Node:** Initializing `dummy = ListNode(0)` allows us to build the merged list without writing special branch logic for the initial head node.
5. **Instant Splicing:** Once either `list1` or `list2` becomes null, we do not need to iterate through the remainder! We simply set `tail.next = list1 if list1 else list2`, attaching the entire remaining chain in $O(1)$ time.

---

### Solution Approach (Step-by-Step)

1. Create a sentinel node: `dummy = ListNode(0)`.
2. Set `tail = dummy`.
3. While `list1` and `list2` are both not null:
   - If `list1.val <= list2.val`:
     - `tail.next = list1`
     - `list1 = list1.next`
   - Else:
     - `tail.next = list2`
     - `list2 = list2.next`
   - `tail = tail.next`
4. Attach remaining nodes: `tail.next = list1 if list1 else list2`.
5. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

```
list1: [1 -> 2 -> 4]
list2: [1 -> 3 -> 4]

dummy -> null, tail = dummy

Step 1:
  list1.val (1) <= list2.val (1)
  tail.next = list1 (1)
  list1 advances to [2 -> 4]
  tail advances to [1]
  Merged: dummy -> [1]

Step 2:
  list1.val (2) > list2.val (1)
  tail.next = list2 (1)
  list2 advances to [3 -> 4]
  tail advances to [1]
  Merged: dummy -> [1 -> 1]

Step 3:
  list1.val (2) <= list2.val (3)
  tail.next = list1 (2)
  list1 advances to [4]
  tail advances to [2]
  Merged: dummy -> [1 -> 1 -> 2]

Step 4:
  list1.val (4) > list2.val (3)
  tail.next = list2 (3)
  list2 advances to [4]
  tail advances to [3]
  Merged: dummy -> [1 -> 1 -> 2 -> 3]

Step 5:
  list1.val (4) <= list2.val (4)
  tail.next = list1 (4)
  list1 becomes null!
  tail advances to [4]
  Merged: dummy -> [1 -> 1 -> 2 -> 3 -> 4]

Step 6:
  list1 is null! Loop terminates.
  Instant attach: tail.next = list2 ([4]).
  Merged: dummy -> [1 -> 1 -> 2 -> 3 -> 4 -> 4]

Return dummy.next.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Interleaved Sorted Lists
- **Input:** `list1 = [1, 2, 4]`, `list2 = [1, 3, 4]`
- **Output:** `[1, 1, 2, 3, 4, 4]`

#### Example 2: Both Lists Empty
- **Input:** `list1 = []`, `list2 = []`
- **Output:** `[]`

#### Example 3: One List Empty
- **Input:** `list1 = []`, `list2 = [0]`
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
    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        tail = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
            
        tail.next = list1 if list1 else list2
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
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;

        while (list1 != nullptr && list2 != nullptr) {
            if (list1->val <= list2->val) {
                tail->next = list1;
                list1 = list1->next;
            } else {
                tail->next = list2;
                list2 = list2->next;
            }
            tail = tail->next;
        }

        tail->next = (list1 != nullptr) ? list1 : list2;
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
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                tail.next = list1;
                list1 = list1.next;
            } else {
                tail.next = list2;
                list2 = list2.next;
            }
            tail = tail.next;
        }

        tail.next = (list1 != null) ? list1 : list2;
        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N + M)$ where $N$ and $M$ are the lengths of `list1` and `list2`. Each step splices one node, traversing at most $N + M$ nodes.
- **Space Complexity:** $O(1)$ strictly in-place — Only pointer links (`next`) are rearranged with zero newly allocated list nodes.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Two-Pointer In-Place Splice with Sentinel/Dummy Head.
- **Trap:** Iterating through remaining nodes element-by-element after one list exhausts. In an array merge, remaining elements must be copied. In a linked list, setting `tail.next = list1 if list1 else list2` instantly splices the entire remainder in $O(1)$!