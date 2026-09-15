---
date: "2026-09-15"
type: leetcode-solution
category: "Linked List"
folder: "07. Linked List"
title: "LeetCode 287: Find the Duplicate Number"
tags:
  - leetcode
  - coding
  - linked-list
  - two-pointers
  - floyds-tortoise-and-hare
  - cycle-detection
  - amazon
  - google
---

# LeetCode 287: Find the Duplicate Number

**Target Companies:** Amazon (Signature Classic), Google, Microsoft, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Array Functional Graph Cycle Entry Detection via Floyd's Tortoise and Hare  

---

### Problem Statement

Given an array of integers `nums` containing $n + 1$ integers where each integer is in the range $[1, n]$ inclusive.

There is only **one repeated number** in `nums`, return this repeated number.

You must solve the problem **without** modifying the array `nums` and uses only constant extra space $O(1)$.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` (length $n + 1$, values $\in [1, n]$)
- **Output:** `int` (the duplicated number)
- **Constraints:**
  - $1 \le n \le 10^5$
  - $\text{nums.length} == n + 1$
  - $1 \le \text{nums}[i] \le n$
  - All the integers in `nums` appear only **once** except for **precisely one integer** which appears **two or more times**.
  - Must not modify `nums`.
  - Must use only $O(1)$ auxiliary space.

---

### Key Idea & Intuition

Because values are in $[1, n]$ and indices range from $0$ to $n$:
- We can view the array as a **directed functional graph** (or implicit linked list) where each index $i$ points to node $nums[i]$:
  $$i \longrightarrow nums[i]$$
- Because $nums[i] \ge 1$, index $0$ has no incoming edge (in-degree is 0), meaning index $0$ is guaranteed to be a valid starting point outside any cycle.
- By the **Pigeonhole Principle**, with $n + 1$ numbers taking values in $[1, n]$, at least one number appears more than once.
- If a number $D$ appears at two different indices $i$ and $j$, both $i \to D$ and $j \to D$. Node $D$ has multiple incoming edges (in-degree $\ge 2$), forming the **entrance of a cycle**!
- Finding the duplicate number is mathematically identical to finding the **entry point of a cycle** in a linked list (LeetCode 142).

#### Mathematical Proof of Cycle Entrance Detection:
Let:
- $L$ = distance from start (node 0) to cycle entrance.
- $d$ = distance from cycle entrance to the meeting point inside the cycle.
- $C$ = cycle length.

When `slow` and `fast` collide:
- `slow` traveled distance: $D_{\text{slow}} = L + d$.
- `fast` traveled distance: $D_{\text{fast}} = L + d + k \cdot C$ (where $k \ge 1$ is full loops).
- Since `fast` moves at twice the speed of `slow`:
  $$2(L + d) = L + d + k \cdot C \implies L + d = k \cdot C \implies L = k \cdot C - d = (k - 1)C + (C - d)$$
This proves that starting one pointer at node $0$ and another pointer at the collision point, and advancing both **one step at a time**, they will meet after traveling exactly $L$ steps at the **cycle entrance**!

---

### Solution Approach (Step-by-Step)

1. **Phase 1 (Cycle Detection):**
   - Initialize `slow = nums[0]`, `fast = nums[0]`.
   - Loop:
     - `slow = nums[slow]` (1 step)
     - `fast = nums[nums[fast]]` (2 steps)
     - If `slow == fast`: break.
2. **Phase 2 (Cycle Entrance Finding):**
   - Reset `ptr1 = nums[0]` (or index 0 with `ptr1 = 0`).
   - `ptr2 = slow`.
   - While `ptr1 != ptr2`:
     - `ptr1 = nums[ptr1]`
     - `ptr2 = nums[ptr2]`
3. Return `ptr1`.

---

### Visual Algorithm Walkthrough

```
nums = [1, 3, 4, 2, 2]
Indices: 0, 1, 2, 3, 4

Graph Transitions (index -> nums[index]):
0 -> 1 -> 3 -> 2 -> 4
               ^    |
               |----|  (Cycle between 2 and 4! Entrance is 2)

Phase 1: Collision Detection
  slow = nums[0] = 1, fast = nums[0] = 1
  Step 1:
    slow = nums[1] = 3
    fast = nums[nums[1]] = nums[3] = 2
  Step 2:
    slow = nums[3] = 2
    fast = nums[nums[2]] = nums[4] = 4
  Step 3:
    slow = nums[2] = 4
    fast = nums[nums[4]] = nums[4] = 4
  slow == fast == 4 (Collision point is node 4!)

Phase 2: Find Entrance
  ptr1 = 0, ptr2 = 4
  Step 1:
    ptr1 = nums[0] = 1
    ptr2 = nums[4] = 2
  Step 2:
    ptr1 = nums[1] = 3
    ptr2 = nums[2] = 4
  Step 3:
    ptr1 = nums[3] = 2
    ptr2 = nums[4] = 2
  ptr1 == ptr2 == 2!

Cycle entrance found: Duplicate number is 2!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Single Duplicate
- **Input:** `nums = [1, 3, 4, 2, 2]`
- **Output:** `2`

#### Example 2: Duplicate Appears More Than Twice
- **Input:** `nums = [3, 1, 3, 4, 2]`
- **Trace:**
  - $0 \to 3 \to 4 \to 2 \to 3$ (Cycle: $3 \to 4 \to 2 \to 3$, entrance is 3).
- **Output:** `3`

#### Example 3: Duplicate is 1
- **Input:** `nums = [1, 1]`
- **Trace:** $0 \to 1 \to 1$ (Self-loop at 1, entrance is 1).
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Phase 1: Detect cycle intersection using fast & slow pointers
        slow = nums[0]
        fast = nums[0]
        
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
                
        # Phase 2: Find cycle entrance
        ptr1 = nums[0]
        ptr2 = slow
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
            
        return ptr1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int findDuplicate(std::vector<int>& nums) {
        // Phase 1: Detect cycle
        int slow = nums[0];
        int fast = nums[0];

        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        // Phase 2: Locate cycle entrance
        int ptr1 = nums[0];
        int ptr2 = slow;
        while (ptr1 != ptr2) {
            ptr1 = nums[ptr1];
            ptr2 = nums[ptr2];
        }

        return ptr1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findDuplicate(int[] nums) {
        // Phase 1: Fast and slow pointer collision
        int slow = nums[0];
        int fast = nums[0];

        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);

        // Phase 2: Locate entrance to the cycle
        int ptr1 = nums[0];
        int ptr2 = slow;

        while (ptr1 != ptr2) {
            ptr1 = nums[ptr1];
            ptr2 = nums[ptr2];
        }

        return ptr1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Phase 1 takes at most $2N$ steps to collide inside the cycle. Phase 2 takes $L \le N$ steps to reach the entrance. Total steps bounded by $3N = O(N)$.
- **Space Complexity:** $O(1)$ strictly constant extra space without modifying the input array.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Implicit Linked List Cycle Detection via Array Value-as-Next Mapping ($i \to nums[i]$).
- **Trap:** Why can't we start `ptr1` at `0`?
  - In our implementation, we start with `slow = nums[0], fast = nums[0]` (already 1 step from 0).
  - So starting `ptr1 = nums[0]` and `ptr2 = slow` correctly mirrors the offset.
  - Alternatively: starting `ptr1 = 0` and `ptr2 = slow` after running `do { slow = nums[slow]; fast = nums[nums[fast]]; } while (slow != fast)` also works if pointers are consistent.