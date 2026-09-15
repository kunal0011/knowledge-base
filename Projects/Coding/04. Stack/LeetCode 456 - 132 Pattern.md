---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 456: 132 Pattern"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - amazon
  - google
---

# LeetCode 456: 132 Pattern

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Monotonic Stack / Subsequence Pattern Matching

---

### Problem Statement

Given an array of `n` integers `nums`, a **132 pattern** is a subsequence of three integers `nums[i]`, `nums[j]` and `nums[k]` such that:
$$i < j < k \quad \text{and} \quad nums[i] < nums[k] < nums[j]$$

Return `true` *if there is a 132 pattern in `nums`, otherwise, return `false`*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $n = \text{len}(nums)$.
- **Output:**
  - `bool`: `True` if any 132 pattern subsequence exists, `False` otherwise.
- **Constraints:**
  - $n == nums.length$
  - $1 \le n \le 2 \times 10^5$
  - $-10^9 \le nums[i] \le 10^9$

---

### Key Idea & Intuition

In a 132 pattern:
- `nums[i]` is the **smallest** element (role "1").
- `nums[j]` is the **peak / largest** element (role "3").
- `nums[k]` is the **middle** element (role "2"), located to the right of `nums[j]` with $nums[k] < nums[j]$.

A brute-force search over all triplets takes $\mathcal{O}(N^3)$, and fixing $j$ while tracking $\min(nums[0 \dots j-1])$ takes $\mathcal{O}(N^2)$. To achieve optimal $\mathcal{O}(N)$ time, we reverse our perspective:

#### Scan Right to Left with a Monotonic Decreasing Stack:
1. Scan from right to left ($i = n-1 \dots 0$).
2. The current element $nums[i]$ serves as a candidate for the smallest value `nums[i]` ("1").
3. We maintain a variable `third` (initialized to $-\infty$), representing the **largest possible value for `nums[k]` ("2")** seen so far that has a known larger peak `nums[j]` ("3") to its left in the scanned portion.
4. The stack maintains candidate peaks `nums[j]` in **monotonically decreasing order** from bottom to top.
5. While scanning leftwards, if $nums[i] < third$, we are done! We have found an $i < k$ such that $nums[i] < third$ where `third` was previously popped by a strictly larger peak $nums[j] > third$. Hence, return `true`.
6. Otherwise, if $nums[i] > \text{stack.top()}$, then $nums[i]$ can act as a new peak $nums[j]$. We pop all elements smaller than $nums[i]$ from the stack and update `third = popped` (since we want `third` to be as large as possible to maximize the chance of finding a subsequent $nums[i] < third$).
7. Push $nums[i]$ onto the stack.

---

### Solution Approach (Step-by-Step)

1. Initialize `stack = []` and `third = -inf`.
2. Loop $i$ backwards from $n - 1$ down to $0$:
   - If $nums[i] < third$:
     - A valid 132 pattern is confirmed $\implies$ return `True`.
   - While `stack` is non-empty and $nums[i] > stack[-1]$:
     - `third = stack.pop()`
   - Push $nums[i]$ onto `stack`.
3. If the loop completes without finding a valid triplet, return `False`.

---

### Visual Algorithm Walkthrough

Let `nums = [3, 1, 4, 2]`:

```
Indices:    0   1   2   3
Values:     3   1   4   2

Scanning backwards:
-------------------------------------------------------------------------
Step 1: i = 3, nums[3] = 2
  nums[3] (2) < third (-inf)? No.
  stack is empty -> push 2.
  Stack: [2]
  third = -inf

-------------------------------------------------------------------------
Step 2: i = 2, nums[2] = 4
  nums[2] (4) < third (-inf)? No.
  stack.top() is 2. 4 > 2!
  Pop 2 -> third = 2.
  Stack is now empty. Push 4.
  Stack: [4]
  third = 2 (We have secured a peak 4 with a right-hand '2'!)

-------------------------------------------------------------------------
Step 3: i = 1, nums[1] = 1
  nums[1] (1) < third (2)? YES! 1 < 2 < 4!
  Pattern found: nums[1]=1 (1), nums[2]=4 (3), nums[3]=2 (2)
  Return True immediately!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard 132 Pattern Found

- **Input:** `nums = [3, 1, 4, 2]`
- **Output:** `true` (triplet `1, 4, 2`)

#### Example 2: Negative and Monotonically Decreasing Array

- **Input:** `nums = [-1, 3, 2, 0]`
- **Tracing:**
  - $i=3$ (`0`): push 0.
  - $i=2$ (`2`): pop 0 $\implies third = 0$, push 2.
  - $i=1$ (`3`): pop 2 $\implies third = 2$, push 3.
  - $i=0$ (`-1`): $-1 < third$ ($2$) $\implies$ `true`.
- **Output:** `true`

#### Example 3: Monotonically Sorted (Ascending / Descending)

- **Input:** `nums = [1, 2, 3, 4]`
  - No peak followed by smaller element. Return `false`.
- **Input:** `nums = [4, 3, 2, 1]`
  - No element smaller than subsequent element. Return `false`.

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        stack = []  # Monotonic decreasing stack storing candidates for nums[j]
        third = float('-inf')  # Candidate for nums[k] ("2")

        # Scan backwards from right to left
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] < third:
                return True

            while stack and nums[i] > stack[-1]:
                third = stack.pop()

            stack.append(nums[i])

        return False
```

#### C++17

```cpp
#include <vector>
#include <stack>
#include <climits>

class Solution {
public:
    bool find132pattern(const std::vector<int>& nums) {
        std::vector<int> stack;
        int third = INT_MIN;
        int n = static_cast<int>(nums.size());

        for (int i = n - 1; i >= 0; --i) {
            if (nums[i] < third) {
                return true;
            }

            while (!stack.empty() && nums[i] > stack.back()) {
                third = stack.back();
                stack.pop_back();
            }

            stack.push_back(nums[i]);
        }

        return false;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public boolean find132pattern(int[] nums) {
        Deque<Integer> stack = new ArrayDeque<>();
        int third = Integer.MIN_VALUE;
        int n = nums.length;

        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] < third) {
                return true;
            }

            while (!stack.isEmpty() && nums[i] > stack.peek()) {
                third = stack.pop();
            }

            stack.push(nums[i]);
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Scanning $N$ elements from right to left.
  - Every element is pushed onto the stack once and popped at most once.
  - Total Time: strictly $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - In the worst case (monotonically increasing array), the stack stores up to $N$ elements.
  - Total Space: $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Why Scanning Right-to-Left Is Superior:**
   - Scanning left-to-right requires fixing the middle element and maintaining a balanced BST or two monotonic structures. Scanning right-to-left simplifies the problem to a single scalar `third` and a single monotonic stack.
2. **Greedy Maximization of `third`:**
   - Every time $nums[i] > stack.top()$, updating `third = stack.pop()` always chooses the largest possible value for $nums[k]$ that has already passed a higher peak. Maximizing `third` maximizes our ability to satisfy $nums[i] < third$.
3. **Handling `INT_MIN` in Languages with Fixed-Width Integers:**
   - In Java and C++, initializing `third = Integer.MIN_VALUE` or `INT_MIN` is safe because no array value can be strictly less than `INT_MIN`. If an array contains `INT_MIN`, it won't prematurely trigger `nums[i] < third`.