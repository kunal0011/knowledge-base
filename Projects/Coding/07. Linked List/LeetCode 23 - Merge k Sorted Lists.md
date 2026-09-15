---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 23: Merge k Sorted Lists"
tags:
  - leetcode
  - coding
  - linked-list
  - divide-and-conquer
  - heap
  - amazon
  - google
---

# LeetCode 23: Merge k Sorted Lists

**Target Companies:** Amazon (Signature Hard #1), Google, Meta, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** Divide and Conquer / Min-Heap Multi-Way Merging  

---

### Problem Statement

You are given an array of $k$ linked-lists `lists`, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

---

### Input & Output Formats & Constraints

- **Input:** `lists: List[Optional[ListNode]]`
- **Output:** `Optional[ListNode]` (head of merged sorted list)
- **Constraints:**
  - $k == \text{lists.length}$
  - $0 \le k \le 10^4$
  - $0 \le \text{lists}[i].\text{length} \le 500$
  - $-10^4 \le \text{lists}[i][j] \le 10^4$
  - $\text{lists}[i]$ is sorted in ascending order.
  - The sum of $\text{lists}[i].\text{length}$ will not exceed $10^4$.

---

### Key Idea & Intuition

Let $N$ be the total number of nodes across all $k$ lists.
- **Naive Sequential Merge:** Merging list 1 with list 2, then with list 3, etc., takes $O(k \cdot N)$ time. For $k = 10^4, N = 10^4$, this is $10^8$ operations and will TLE.

There are two optimal approaches:

#### Approach 1: Divide and Conquer (Pairwise Bottom-Up Merge) — $O(N \log k)$ Time, $O(1)$ Space
Instead of merging one list at a time sequentially, we pair adjacent lists and merge them two-by-two:
- Round 1: Merge pairs $(0, 1), (2, 3), \dots \to \frac{k}{2}$ lists remain.
- Round 2: Merge remaining pairs $\to \frac{k}{4}$ lists remain.
- There are exactly $\lceil \log_2 k \rceil$ rounds. In each round, every node is touched once, costing $O(N)$.
- Total Time: $O(N \log k)$.
- Total Space: $O(1)$ auxiliary space if done iteratively in-place.

#### Approach 2: Priority Queue (Min-Heap) — $O(N \log k)$ Time, $O(k)$ Space
- Insert the head of each non-empty list into a min-heap of size at most $k$.
- Repeatedly extract the minimum node, append it to the merged list, and if that node has a `next`, push `next` into the heap.
- Each extraction/insertion costs $O(\log k)$, performed $N$ times $\implies O(N \log k)$.

---

### Solution Approach (Step-by-Step — Divide & Conquer)

1. Check base case: if `lists` is empty, return `None`.
2. While `len(lists) > 1`:
   - Initialize `merged_lists = []`.
   - Iterate through `lists` with step 2:
     - `l1 = lists[i]`
     - `l2 = lists[i + 1]` if $i + 1 < \text{len}(lists)$ else `None`.
     - `merged_lists.append(mergeTwoLists(l1, l2))`.
   - Update `lists = merged_lists`.
3. Return `lists[0]`.

---

### Visual Algorithm Walkthrough

```
Input: 4 lists: L0, L1, L2, L3 (Total nodes = N)

Round 1: Pairwise merge (step = 1)
   L0 -------\
              +----> Merge(L0, L1) ======> M01  (N/2 nodes touched)
   L1 -------/

   L2 -------\
              +----> Merge(L2, L3) ======> M23  (N/2 nodes touched)
   L3 -------/
   Time for Round 1 = O(N)

Round 2: Merge intermediate pairs (step = 2)
   M01 ------\
              +----> Merge(M01, M23) ====> Final Merged List (N nodes)
   M23 ------/
   Time for Round 2 = O(N)

Total Rounds = log2(4) = 2.
Total Time = 2 * O(N) = O(N log k).
No extra nodes allocated!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard 3-List Merge
- **Input:** `lists = [[1, 4, 5], [1, 3, 4], [2, 6]]`
- **Trace:**
  - Round 1: `merge([1, 4, 5], [1, 3, 4])` $\to$ `[1, 1, 3, 4, 4, 5]`. Keep `[2, 6]`.
  - Round 2: `merge([1, 1, 3, 4, 4, 5], [2, 6])` $\to$ `[1, 1, 2, 3, 4, 4, 5, 6]`.
- **Output:** `[1, 1, 2, 3, 4, 4, 5, 6]`

#### Example 2: Empty Array of Lists
- **Input:** `lists = []`
- **Output:** `[]`

#### Example 3: Array of Empty Lists
- **Input:** `lists = [[]]`
- **Output:** `[]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Divide & Conquer)
```python
from typing import List, Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return None
            
        def merge_two(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode(0)
            tail = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
                tail = tail.next
            tail.next = l1 if l1 else l2
            return dummy.next
            
        while len(lists) > 1:
            merged: List[Optional[ListNode]] = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                merged.append(merge_two(l1, l2))
            lists = merged
            
        return lists[0]
```

#### 2. C++ (C++17 / STL — Min-Heap & Divide-and-Conquer)
```cpp
#include <vector>
#include <queue>

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeKLists(std::vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;

        auto mergeTwo = [](ListNode* l1, ListNode* l2) -> ListNode* {
            ListNode dummy(0);
            ListNode* tail = &dummy;
            while (l1 != nullptr && l2 != nullptr) {
                if (l1->val <= l2->val) {
                    tail->next = l1;
                    l1 = l1->next;
                } else {
                    tail->next = l2;
                    l2 = l2->next;
                }
                tail = tail->next;
            }
            tail->next = (l1 != nullptr) ? l1 : l2;
            return dummy.next;
        };

        int n = lists.size();
        int interval = 1;
        while (interval < n) {
            for (int i = 0; i + interval < n; i += interval * 2) {
                lists[i] = mergeTwo(lists[i], lists[i + interval]);
            }
            interval *= 2;
        }

        return lists[0];
    }
};
```

#### 3. Java (Modern, Typed — Min-Heap PriorityQueue)
```java
import java.util.PriorityQueue;

class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> Integer.compare(a.val, b.val));
        for (ListNode node : lists) {
            if (node != null) {
                pq.offer(node);
            }
        }

        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;

        while (!pq.isEmpty()) {
            ListNode smallest = pq.poll();
            tail.next = smallest;
            tail = tail.next;

            if (smallest.next != null) {
                pq.offer(smallest.next);
            }
        }

        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log k)$ where $N$ is the total count of nodes and $k$ is the number of linked lists. There are $\log k$ levels of merging, and each node is inspected once per level.
- **Space Complexity:**
  - **Divide & Conquer:** $O(1)$ auxiliary space if doing in-place array intervals (or $O(k)$ for the intermediate vector).
  - **Priority Queue:** $O(k)$ auxiliary space for the heap storing up to $k$ list heads.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Divide and Conquer Bottom-Up Pairwise Merge Tree.
- **Trap:** Comparing `ListNode` objects directly in Python's `heapq` without a custom wrapper or tuple `(node.val, id(node), node)`: in Python, if node values are equal, `heapq` attempts to compare `ListNode` with `ListNode`, raising a `TypeError: '<' not supported between instances of 'ListNode'`. Divide and Conquer avoids this issue entirely.