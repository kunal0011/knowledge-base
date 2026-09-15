---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 70: Climbing Stairs"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - fibonacci
  - math
  - amazon
  - google
  - meta
---

# LeetCode 70: Climbing Stairs

**Target Companies:** Amazon (Top #1), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** 1D Dynamic Programming / Recurrence Relations / Space Optimization  

---

### Problem Statement

You are climbing a staircase. It takes `n` steps to reach the top.

Each time you can either climb `1` or `2` steps. In how many distinct ways can you climb to the top?

---

### Input & Output Formats & Constraints

- **Input:** `n: int` — Total number of steps to reach the top.
- **Output:** `int` — Count of distinct combinations of 1-step and 2-step climbs.
- **Constraints:**
  - $1 \le n \le 45$

---

### Key Idea & Intuition

1. **Optimal Substructure:**
   - To land on step $i$, your preceding move was either:
     - A 1-step climb from step $i - 1$.
     - A 2-step climb from step $i - 2$.
   - Since these two incoming branches are mutually exclusive and exhaustive:
     $$\text{dp}[i] = \text{dp}[i - 1] + \text{dp}[i - 2]$$
   - This matches the canonical **Fibonacci sequence**:
     - $\text{dp}[1] = 1$ (one 1-step climb: `[1]`)
     - $\text{dp}[2] = 2$ (two combinations: `[1, 1]`, `[2]`)
     - $\text{dp}[3] = 3$ (`[1, 1, 1]`, `[1, 2]`, `[2, 1]`)

2. **Space Optimization ($\mathcal{O}(1)$ Space):**
   - Because calculating $\text{dp}[i]$ only requires the two immediately preceding values, maintaining an entire array of size $n + 1$ is unnecessary.
   - Using two rolling integer variables `prev2` and `prev1` achieves $\mathcal{O}(1)$ space complexity.

---

### Solution Approach (Step-by-Step)

1. **Handle Base Cases:**
   - If $n \le 2$, return $n$.
2. **Rolling Variables Initialization:**
   - `prev2 = 1` ($\text{dp}[1]$)
   - `prev1 = 2` ($\text{dp}[2]$)
3. **Iterate from Step 3 to $n$:**
   - For $i$ from $3$ to $n$:
     - `curr = prev1 + prev2`
     - `prev2 = prev1`
     - `prev1 = curr`
4. **Return:**
   - Return `prev1`.

---

### Visual Algorithm Walkthrough

For $n = 5$:

```
Step 1: 1 way   ([1])
Step 2: 2 ways  ([1,1], [2])

i = 3:
  curr = prev1 + prev2 = 2 + 1 = 3  ([1,1,1], [1,2], [2,1])
  prev2 = 2, prev1 = 3

i = 4:
  curr = prev1 + prev2 = 3 + 2 = 5  ([1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2])
  prev2 = 3, prev1 = 5

i = 5:
  curr = prev1 + prev2 = 5 + 3 = 8
  prev2 = 5, prev1 = 8

Total unique ways for n = 5: 8.
```

---

### Solved Examples with Multiple Inputs

| Case | `n` | Intermediate Sequence | Result | Explanation |
|---|---|---|---|---|
| **Base Case 1** | `1` | `[1]` | `1` | Only step 1 |
| **Base Case 2** | `2` | `[1, 1]`, `[2]` | `2` | Two ways |
| **Small 3** | `3` | $1 + 2 = 3$ | `3` | Three ways |
| **Standard 5** | `5` | $1, 2, 3, 5, 8$ | `8` | Fibonacci progression |
| **Upper Bound 45** | `45` | Linear progression | `1836311903` | Fits within 32-bit signed int ($< 2 \times 10^9$) |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — $\mathcal{O}(1)$ Space)
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
            
        prev2, prev1 = 1, 2
        
        for _ in range(3, n + 1):
            curr = prev1 + prev2
            prev2 = prev1
            prev1 = curr
            
        return prev1
```

#### 2. C++ (C++17 / STL — $\mathcal{O}(1)$ Space)
```cpp
class Solution {
public:
    int climbStairs(int n) {
        if (n <= 2) return n;

        int prev2 = 1;
        int prev1 = 2;

        for (int i = 3; i <= n; ++i) {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
};
```

#### 3. Java (Modern, Typed — $\mathcal{O}(1)$ Space)
```java
class Solution {
    public int climbStairs(int n) {
        if (n <= 2) return n;

        int prev2 = 1;
        int prev1 = 2;

        for (int i = 3; i <= n; i++) {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$  
  The single loop executes $n - 2$ times for $n \ge 3$, performing $\mathcal{O}(1)$ additions per iteration. For $n = 45$, finishes in $< 0.1$ ms.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only two integer tracking variables (`prev2`, `prev1`) are maintained in registers.

---

### Takeaway Pattern & Interview Traps

1. **Integer Overflow Check:**
   - Notice the constraint $n \le 45$. $F(45) = 1,836,311,903$, which is just under `INT_MAX` ($2,147,483,647$).
   - If $n$ were $46$, $F(46) = 2,971,215,073$, which would overflow standard 32-bit signed integers.
2. **Step Step Variations:**
   - If the problem allowed taking $1, 2$, or $3$ steps (Tribonacci), the recurrence expands to $\text{dp}[i] = \text{dp}[i-1] + \text{dp}[i-2] + \text{dp}[i-3]$ with three rolling variables.