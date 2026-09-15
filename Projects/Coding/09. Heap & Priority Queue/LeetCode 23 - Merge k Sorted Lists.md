---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 23: Merge k Sorted Lists"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - linked-list
  - divide-and-conquer
  - amazon
  - google
---

# LeetCode 23: Merge k Sorted Lists

**Target Companies:** Amazon (Top Classic), Google, Microsoft, Meta, Apple, Bloomberg  
**Difficulty:** Hard  
**Topic:** Priority Queue (Min-Heap) / Divide & Conquer / K-Way Merge

---

### Problem Statement

You are given an array of `k` linked-lists `lists`, each linked-list is sorted in ascending order.

*Merge all the linked-lists into one sorted linked-list and return it.*

---

### Input & Output Formats & Constraints

- **Input:**
  - `lists`: `List[Optional[ListNode]]`, an array of $k$ sorted singly-linked lists.
- **Output:**
  - `Optional[ListNode]`: Head of the single merged sorted linked list.
- **Constraints:**
  - $k == \text{lists.length}$
  - $0 \le k \le 10^4$
  - $0 \le \text{lists}[i]\text{.length} \le 500$
  - $-10^4 \le \text{lists}[i][j] \le 10^4$
  - $\text{lists}[i]$ is sorted in ascending order.
  - The total number of nodes across all lists is in the range $[0, 10^4]$.

---

### Key Idea & Intuition

Because each of the $k$ lists is already sorted, the globally smallest remaining node at any step must be among the **current front nodes of the $k$ lists**.

This problem is the canonical **$K$-Way Merge**, solvable with two principal paradigms:

#### Paradigm 1: Min-Heap / Priority Queue ($\mathcal{O}(N \log k)$ Time, $\mathcal{O}(k)$ Space)
1. Initialize a min-heap with the head node of each non-empty linked list.
2. In Python, store tuples `(node.val, list_index, node)` to ensure unique, comparable keys when node values tie.
3. In each iteration:
   - Extract the minimum node from the heap: `(val, idx, node) = heappop(min_heap)`.
   - Append `node` to the merged list.
   - If `node.next` exists, push `(node.next.val, idx, node.next)` into the heap.
4. Continue until the heap is empty.
5. Total nodes $N$, heap size at most $k \implies \mathcal{O}(N \log k)$ time.

#### Paradigm 2: Divide and Conquer ($\mathcal{O}(N \log k)$ Time, $\mathcal{O}(1)$ Auxiliary Space)
1. Pair up the $k$ lists and merge each pair using the standard 2-list merge algorithm (LeetCode 21).
2. After the first round, $k/2$ lists remain; after the second, $k/4$, and so on.
3. Repeat $\lceil \log_2 k \rceil$ times until a single merged list remains.
4. Requires zero heap allocations.

---

### Solution Approach (Step-by-Step)

#### Priority Queue Algorithm:
1. Create a sentinel `dummy = ListNode(0)` and `curr = dummy`.
2. Initialize `min_heap = []`.
3. For each list head at index $i$:
   - If `lists[i]` is not null:
     - Push `(lists[i].val, i, lists[i])` onto `min_heap`.
4. While `min_heap` is non-empty:
   - `val, i, node = heappop(min_heap)`
   - `curr.next = node`
   - `curr = curr.next`
   - If `node.next` is not null:
     - `heappush(min_heap, (node.next.val, i, node.next))`
5. Return `dummy.next`.

---

### Visual Algorithm Walkthrough

Let `lists = [L0: 1->4->5, L1: 1->3->4, L2: 2->6]`:

```
Initial Heap (heads of L0, L1, L2):
  Heap: [ (1, 0, L0), (1, 1, L1), (2, 2, L2) ]

Step 1:
  Pop (1, 0, L0). Append 1.
  L0.next is 4. Push (4, 0, L0_next).
  Heap: [ (1, 1, L1), (2, 2, L2), (4, 0, L0) ]
  Merged: 1 ->

Step 2:
  Pop (1, 1, L1). Append 1.
  L1.next is 3. Push (3, 1, L1_next).
  Heap: [ (2, 2, L2), (4, 0, L0), (3, 1, L1) ]
  Merged: 1 -> 1 ->

Step 3:
  Pop (2, 2, L2). Append 2.
  L2.next is 6. Push (6, 2, L2_next).
  Heap: [ (3, 1, L1), (4, 0, L0), (6, 2, L2) ]
  Merged: 1 -> 1 -> 2 ->

Step 4:
  Pop (3, 1, L1). Append 3.
  Push 4 from L1.
  Merged: 1 -> 1 -> 2 -> 3 ->

Step 5:
  Pop 4 (from L0). Append 4. Push 5.
Step 6:
  Pop 4 (from L1). Append 4. L1 exhausted.
Step 7:
  Pop 5 (from L0). Append 5. L0 exhausted.
Step 8:
  Pop 6 (from L2). Append 6. L2 exhausted.

Result: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard 3 Lists

- **Input:** `lists = [[1,4,5],[1,3,4],[2,6]]`
- **Output:** `[1,1,2,3,4,4,5,6]`

#### Example 2: Empty Array of Lists

- **Input:** `lists = []`
- **Output:** `[]` (Returns `null`)

#### Example 3: Array of Empty Lists

- **Input:** `lists = [[]]`
- **Output:** `[]` (Returns `null`)

---

### Multi-Language Implementations

#### Python 3

```python
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val: int = 0, next: Optional['ListNode'] = None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap = []

        # Push the head of each non-empty list
        # Using (node.val, i, node) to safely break ties without node comparisons
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(min_heap, (head.val, i, head))

        dummy = ListNode(0)
        curr = dummy

        while min_heap:
            val, i, node = heapq.heappop(min_heap)
            curr.next = node
            curr = curr.next

            if node.next:
                heapq.heappush(min_heap, (node.next.val, i, node.next))

        return dummy.next
```

#### C++17

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
private:
    struct Compare {
        bool operator()(const ListNode* a, const ListNode* b) const {
            return a->val > b->val; // Min-heap comparator
        }
    };

public:
    ListNode* mergeKLists(std::vector<ListNode*>& lists) {
        std::priority_queue<ListNode*, std::vector<ListNode*>, Compare> min_heap;

        for (ListNode* head : lists) {
            if (head != nullptr) {
                min_heap.push(head);
            }
        }

        ListNode dummy(0);
        ListNode* curr = &dummy;

        while (!min_heap.empty()) {
            ListNode* smallest = min_heap.top();
            min_heap.pop();

            curr->next = smallest;
            curr = curr->next;

            if (smallest->next != nullptr) {
                min_heap.push(smallest->next);
            }
        }

        return dummy.next;
    }
};
```

#### Java

```java
import java.util.PriorityQueue;

public class Solution {
    public static class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;

        // Min-heap ordered by node values
        PriorityQueue<ListNode> minHeap = new PriorityQueue<>(
            (a, b) -> Integer.compare(a.val, b.val)
        );

        for (ListNode head : lists) {
            if (head != null) {
                minHeap.offer(head);
            }
        }

        ListNode dummy = new ListNode(0);
        ListNode curr = dummy;

        while (!minHeap.isEmpty()) {
            ListNode smallest = minHeap.poll();
            curr.next = smallest;
            curr = curr.next;

            if (smallest.next != null) {
                minHeap.offer(smallest.next);
            }
        }

        return dummy.next;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log k)$
  - Where $N$ is the total number of nodes across all $k$ lists.
  - The heap holds at most $k$ elements. Each push and pop operation takes $\mathcal{O}(\log k)$ time.
  - With $N$ nodes total, overall time is $\mathcal{O}(N \log k)$.
- **Space Complexity:** $\mathcal{O}(k)$
  - The heap contains at most one node from each of the $k$ linked lists simultaneously.

---

### Takeaway Pattern & Interview Traps

1. **Tuple Tie-Breaking in Python:**
   - In Python, `heapq` will compare the next element in a tuple if values match. If you push `(node.val, node)`, Python crashes with `TypeError: '<' not supported between instances of 'ListNode'` when two node values match! Inserting the list index `(node.val, i, node)` guarantees a unique second comparison key.
2. **K-Way Merge Generalization:**
   - This exact same pattern applies to merging $k$ sorted arrays, log aggregation from $k$ server streams, and external merge sort on disk files that exceed RAM.