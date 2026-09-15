---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 457: Circular Array Loop"
tags:
  - leetcode
  - coding
  - two-pointers
  - floyds-tortoise-and-hare
  - cycle-detection
  - google
  - amazon
---

# LeetCode 457: Circular Array Loop

**Target Companies:** Google, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Fast & Slow Pointers / Floyd's Cycle Detection / In-Place Path Marking  

---

### Problem Statement

You are playing a game involving a **circular** array of non-zero integers `nums` of length $n$. Each `nums[i]` represents the number of indices forward/backward you must jump from index $i$:
- If `nums[i] > 0`, move `nums[i]` steps forward.
- If `nums[i] < 0`, move `nums[i]` steps backward.

Because the array is circular, jumping past the last element wraps around to the beginning, and jumping backward past the first element wraps around to the end.

A **valid loop** must satisfy:
1. Every movement in the loop follows the **same direction** (all positive or all negative).
2. The loop contains **more than 1** element (a 1-element loop jumping to itself is invalid).

Return `true` if there is a cycle in `nums`, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `bool` (`True` if valid loop exists, `False` otherwise)
- **Constraints:**
  - $1 \le \text{nums.length} \le 5000$
  - $-1000 \le \text{nums}[i] \le 1000$
  - $\text{nums}[i] \ne 0$

---

### Key Idea & Intuition

This is **Floyd's Tortoise and Hare** algorithm applied on a functional graph with two critical constraints:
1. **Unidirectional Constraint:** Every hop along the path must maintain `nums[curr] * nums[next] > 0`. If a hop flips sign, that path cannot form a valid cycle.
2. **Cycle Length > 1:** If `curr == get_next(curr)`, it's a self-loop (e.g. `nums[i] % n == 0`), which is explicitly forbidden.

#### How to Guarantee $O(N)$ Time Complexity
A naive fast/slow pointer run from each index $i$ could take $O(N^2)$ if paths overlap. To achieve true $O(N)$ time:
- Once a path starting at $i$ is verified to contain no valid loop, we mark all nodes along that trajectory with `0`.
- Because the problem guarantees $\text{nums}[i] \ne 0$, an encountered `0` denotes an already-failed path and can be skipped immediately!

---

### Solution Approach (Step-by-Step)

1. Helper `next_pos(i)`: returns `((i + nums[i]) % n + n) % n`.
2. For each index $i \in [0, n - 1]$:
   - If `nums[i] == 0`, skip (already visited).
   - Initialize `slow = i`, `fast = next_pos(i)`.
   - Direction test: ensure `nums[i] * nums[slow] > 0` and `nums[i] * nums[fast] > 0` and `nums[i] * nums[next_pos(fast)] > 0`.
   - Advance `slow` by 1 step, `fast` by 2 steps while directions remain identical.
   - If `slow == fast`:
     - Check if it's a self-loop: if `slow == next_pos(slow)`, break (invalid).
     - Otherwise, return `true`.
   - If loop terminates without a valid cycle, mark all elements reachable from $i$ in the same direction as `0` to prevent redundant re-exploration.
3. Return `false` if all indices are exhausted.

---

### Visual Algorithm Walkthrough

```
nums = [2, -1, 1, 2, 2], n = 5

Index:     0   1   2   3   4
Value:     2  -1   1   2   2

Start at index 0 (value > 0, forward direction):
- slow = 0 -> next_pos(0) = (0 + 2) % 5 = 2
- fast = next_pos(0) = 2 -> next_pos(2) = (2 + 1) % 5 = 3

Round 1:
  slow is at 2 (val = 1 > 0)
  fast is at 3 (val = 2 > 0)
  Advance slow -> next_pos(2) = 3
  Advance fast -> next_pos(3) = 0, then next_pos(0) = 2
  slow = 3, fast = 2

Round 2:
  Advance slow -> next_pos(3) = 0
  Advance fast -> next_pos(2) = 3, then next_pos(3) = 0
  slow = 0, fast = 0 -> slow == fast! (Cycle Detected!)

Self-loop Check:
  Is slow == next_pos(slow)?
  slow = 0, next_pos(0) = 2. 0 != 2 -> Cycle length > 1.
  Direction: 0 -> 2 (1 > 0) -> 3 (2 > 0) -> 0 (2 > 0). All positive!

Result: Return True!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Valid Forward Cycle
- **Input:** `nums = [2, -1, 1, 2, 2]`
- **Cycle:** $0 \to 2 \to 3 \to 0$ (length 3, all $> 0$).
- **Output:** `true`

#### Example 2: Invalid Self-Loop
- **Input:** `nums = [-1, 2]`
- **Trace:**
  - Index 0: `(-1 + (-1)) % 2 = 0`. Jumps to itself! Cycle length = 1. Invalid.
  - Index 1: `(1 + 2) % 2 = 1`. Jumps to itself! Cycle length = 1. Invalid.
- **Output:** `false`

#### Example 3: Mixed Direction Cycle (Invalid)
- **Input:** `nums = [-2, 1, -1, -2, -2]`
- **Trace:**
  - Cycle involves index 1 (positive) and index 2 (negative).
  - Direction constraint violated.
- **Output:** `false`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        n = len(nums)
        
        def get_next(i: int) -> int:
            return ((i + nums[i]) % n + n) % n
            
        for i in range(n):
            if nums[i] == 0:
                continue
                
            slow = i
            fast = get_next(i)
            direction = nums[i]
            
            # Floyd's Tortoise and Hare
            while (nums[slow] * direction > 0 and 
                   nums[fast] * direction > 0 and 
                   nums[get_next(fast)] * direction > 0):
                if slow == fast:
                    # Self-loop check (cycle length 1)
                    if slow == get_next(slow):
                        break
                    return True
                slow = get_next(slow)
                fast = get_next(get_next(fast))
                
            # Mark all nodes in this invalid path as 0
            curr = i
            while nums[curr] * direction > 0:
                nxt = get_next(curr)
                nums[curr] = 0
                curr = nxt
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    bool circularArrayLoop(std::vector<int>& nums) {
        int n = nums.size();
        
        auto getNext = [&](int i) {
            return ((i + nums[i]) % n + n) % n;
        };

        for (int i = 0; i < n; ++i) {
            if (nums[i] == 0) continue;
            
            int slow = i;
            int fast = getNext(i);
            int direction = nums[i];

            while (nums[slow] * direction > 0 &&
                   nums[fast] * direction > 0 &&
                   nums[getNext(fast)] * direction > 0) {
                if (slow == fast) {
                    if (slow == getNext(slow)) {
                        break; // Self loop
                    }
                    return true;
                }
                slow = getNext(slow);
                fast = getNext(getNext(fast));
            }

            // Mark invalid path as 0
            int curr = i;
            while (nums[curr] * direction > 0) {
                int nxt = getNext(curr);
                nums[curr] = 0;
                curr = nxt;
            }
        }
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean circularArrayLoop(int[] nums) {
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            if (nums[i] == 0) continue;

            int slow = i;
            int fast = getNext(nums, i);
            int direction = nums[i];

            while (nums[slow] * direction > 0 &&
                   nums[fast] * direction > 0 &&
                   nums[getNext(nums, fast)] * direction > 0) {
                if (slow == fast) {
                    if (slow == getNext(nums, slow)) {
                        break; // Self loop of length 1
                    }
                    return true;
                }
                slow = getNext(nums, slow);
                fast = getNext(nums, getNext(nums, fast));
            }

            // Invalidate explored nodes to maintain O(N)
            int curr = i;
            while (nums[curr] * direction > 0) {
                int next = getNext(nums, curr);
                nums[curr] = 0;
                curr = next;
            }
        }
        return false;
    }

    private int getNext(int[] nums, int i) {
        int n = nums.length;
        return ((i + nums[i]) % n + n) % n;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each node is marked as `0` as soon as it is identified as part of a dead-end or non-looping path. Thus, each index is processed a constant number of times.
- **Space Complexity:** $O(1)$ — Modifies `nums` in-place using pointer variables with zero additional memory allocation.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Floyd's Cycle Finding on Dynamic Functional Graphs + In-Place Graph Poisoning (marking dead components with 0).
- **Trap:** Forgetting that modulo in C++ and Java can produce negative numbers when `(i + nums[i]) < 0`. Always use `((i + nums[i]) % n + n) % n`.
- **Trap:** Not checking for 1-node loops (`nums[i] % n == 0`), which are explicitly invalid under the problem statement.